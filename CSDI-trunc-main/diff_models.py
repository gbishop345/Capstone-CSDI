import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer

# creates a transformation encoder for sequential model
def get_torch_trans(heads=8, layers=1, channels=64):
    # heads: number of heads in multi head self attention mechanism
    # layers: number of transformation layers
    # channels: dimension of the input

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )

    # returns nn.TransformerEncoder wrapped layers
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

# more efficient transformer used in linear cases
def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

# creates a 1D convolution layer
def Conv1d_with_init(in_channels, out_channels, kernel_size):
    # in_channels: input size
    # out_channels: output size
    # kernel_seize: size of the convolution kernel

    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight) # initializes convolution weights with kaiming normal intialization
    return layer

# embedding for diffusion model
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        # num_steps: number of time steps for embedding process
        # embedding_dim: dimesion of each embedding step, helps to encode info
        # projection_dim: output size, defaults to embedding_dim

        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer( # this stores the embedding tables
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        # set up the fully connected layers, one for time and one for features
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step] # retrieves the precomputed embedding table
        x = self.projection1(x) # embedding is passed through the first linear projection
        x = F.silu(x) # activation function
        x = self.projection2(x) # passed through second linear projection
        x = F.silu(x) # activation function
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # creates tensor of shape (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # frequency values for embedding
        table = steps * frequencies  # get the total size of the sinusoidal embedding (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # create the actual table (T,dim*2)
        return table


# the deep learning diffusion model
class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        # x: input data
        # cond_indo: additional conditional info
        # diffusion_step: the current step in the diffusion process

        B, inputdim, K, L = x.shape # get the different dimensions of the input data

        x = x.reshape(B, inputdim, K * L) # flatten the K, L dimension for 1D convolution
        x = self.input_projection(x) # convolution layer to transform input to channels
        x = F.relu(x) # non linear activation function
        x = x.reshape(B, self.channels, K, L) # undo the K,L flattening

        diffusion_emb = self.diffusion_embedding(diffusion_step) # get the diffusion embedding

        skip = [] # connections to skip
        for layer in self.residual_layers: # for each layer we need to process x based on cond_info
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        # skip connections and normalize values to balance contribtution of all residual layers. 
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers)) 
        x = x.reshape(B, self.channels, K * L) # flatten K,L for final projection
        x = self.output_projection1(x)  # maps output back to channels (B,channel,K*L)
        x = F.relu(x) # non linear activation function
        x = self.output_projection2(x)  # maps output back to 1 (B,1,K*L)
        x = x.reshape(B, K, L) # reshape back to original input dimensions
        return x

# residual blocks to improve stable model training
class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        # side_dim: diminsion of conditional info
        # channels: number of channels in input feature map
        # diffusion_embedding_dim: dimension of diffusion embedding
        # nheads: number of heads for multi head attention
        # is_linear: boolean indication of whether to use linear attention mechanism or traditional

        super().__init__()
        # projects diffusion embedding (time step info) to match chennel dimension
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        # projects conditional info for 2 times the size for a later split
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)

        # projections for intermediate and final processing, both 2 times the size
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear: # if linear use linear transformation 
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels) 
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else: # if not linear use traditional transformation
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    # foward pass through time dimension
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape # get target shape
        if L == 1: # if time dimension is 1 just return the input unchanged
            return y
        
        # reshape for pass to focus on time dimension by flattening by B,K
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear: # reshape for linear layer
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else: # reshape for traditional layer
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        
        # final reshapeing for processing
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape # get the target shape
        if K == 1: # if feature dimension is 1 return input
            return y
        
        # reshape to focus on feature dimension K by flattining by B,L
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)

        if self.is_linear: # reshape for linear layer
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else: # reshape for traditional layer
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        # final reshaping for processing
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape # extract each dimension of input tensor 
        base_shape = x.shape
        x = x.reshape(B, channel, K * L) # flatten by time and spatial feature

        # project the embedding to match x
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb # add the embedding to the input x

        y = self.forward_time(y, base_shape) # apply time fuction reshaping 
        y = self.forward_feature(y, base_shape)  # apply feature function reshaping (B,channel,K*L)
        y = self.mid_projection(y)  # doubles the channel size for split (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L) # conditional info is reshaped to match y
        cond_info = self.cond_projection(cond_info)  # doubles the channel size (B,2*channel,K*L)
        y = y + cond_info # add the conditional info to y

        gate, filter = torch.chunk(y, 2, dim=1) # split y into the gate and the filter 
        y = torch.sigmoid(gate) * torch.tanh(filter)  # apply activation function to gate and filter (B,channel,K*L)
        y = self.output_projection(y) # projects y back to (B, 2*channel, K*L)

        residual, skip = torch.chunk(y, 2, dim=1) # split y to add the residual and skip connections
        x = x.reshape(base_shape) # reshape back to original (B, channel, K, L)
        residual = residual.reshape(base_shape) # get the residual back to original shape
        skip = skip.reshape(base_shape) # get the skip back to original shape
        return (x + residual) / math.sqrt(2.0), skip # return the x + residual scaled for better residual connection