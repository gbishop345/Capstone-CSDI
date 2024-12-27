import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        # Set up model configuration
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        # Total embedding dimension
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1

        # Embedding layer
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)
        
        # Diffusion model configuration
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        
        # Define input dimensions
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # Initialize the discriminator with input channels matching the diffusion model's output
        input_channels = self.target_dim  # Assuming data has 'target_dim' channels
        hidden_dim = 64  # Default hidden dimension for the discriminator
        temb_dim = self.emb_time_dim  # Use the same temporal embedding dimension
        self.discriminator = Discriminator(input_channels=input_channels, hidden_dim=hidden_dim, temb_dim=temb_dim)

        self.T_trunc = config_diff['T_trunc']
        self.lambda_gan = config['loss']['lambda_gan']
        self.lambda_js = config['loss']['lambda_js']

        # Diffusion process parameters
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        # Precompute alpha_hat and alpha values
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)


    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow( # create divisors
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term) # apply sine to even indicies
        pe[:, :, 1::2] = torch.cos(position * div_term) # apply cos to odd indicies
        return pe 

    # mask some values from the data to have something to inpute
    def get_randmask(self, observed_mask): 
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1) # make it 2D
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item() # how many points are there
            num_masked = round(num_observed * sample_ratio) # how many points need to be masked
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask # returns tensor with some points masked to 0
    
    # get a mask partially from random and partially historical observations
    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask # if no mask is given use the observed one
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand() # use random or historical
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i] 
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask
    
    # applied a mask to the observed data for controlled testing
    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    # generates the side info by combining time and feature embedding
    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape # get shape info

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) # creates time embedding for observed points
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # match the right dimensions
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # creates feature embedding (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) # match the right dimensions

        side_info = torch.cat([time_embed, feature_embed], dim=-1) # add time and feature for side info (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1) # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1) # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info
    
    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.compute_generator_loss(
                observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train, set_t=t
            )
            loss_sum += loss.detach()
        average_loss = loss_sum / self.num_steps
        return average_loss
    
    def stable_softmax(self, tensor, dim=-1):
        tensor_max, _ = torch.max(tensor, dim=dim, keepdim=True)
        tensor_exp = torch.exp(tensor - tensor_max)
        tensor_exp_sum = torch.sum(tensor_exp, dim=dim, keepdim=True)
        softmax = tensor_exp / tensor_exp_sum
        return softmax

    #Compute stable Jensen-Shannon Divergence 
    def js_divergence(self, p, q, eps=1e-8):

        # Add epsilon for numerical stability
        p = p + eps
        q = q + eps

        # Apply softmax to convert to probability distributions
        p = self.stable_softmax(p, dim=-1)
        q = self.stable_softmax(q, dim=-1)

        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)

        # Compute the midpoint
        m = 0.5 * (p + q)
        js_div = 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') + F.kl_div(q.log(), m, reduction='batchmean'))
        return js_div

    def compute_generator_loss(self, observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train, set_t=-1):
        B, K, L = observed_data.shape

        # Select a specific time step for validation, otherwise sample randomly
        '''
        this part to sample T_trunc more often is new
        '''
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            # Adjust the probability distribution to oversample T_trunc - 1
            t_probs = torch.ones(self.T_trunc)
            oversampling_factor = 2  # Adjust this factor as needed
            t_probs[-1] *= oversampling_factor  # Increase probability for T_trunc - 1
            t_probs = t_probs / t_probs.sum()
            t = torch.multinomial(t_probs, B, replacement=True).to(self.device)

        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)

        # Generate noisy data
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        # Predict noise using the model
        predicted = self.diffmodel(total_input, side_info, t)

        # Calculate the loss for noise prediction at each step
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        # Apply GAN loss and JS divergence at T_trunc
        '''
        all loss function calulation at T_trunc is new
        '''
        if (t == self.T_trunc - 1).any():
            t_trunc_indices = (t == self.T_trunc - 1).nonzero(as_tuple=True)[0]

            # Get alpha_t for the indices at T_trunc
            alpha_t = self.alpha_torch[t[t_trunc_indices]].squeeze(-1).squeeze(-1)
            sqrt_alpha_t = torch.sqrt(alpha_t)[:, None, None]
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)[:, None, None]

            # Fake data at T_trunc
            fake_data_at_T_trunc = (noisy_data[t_trunc_indices] - sqrt_one_minus_alpha_t * predicted[t_trunc_indices]) / sqrt_alpha_t

            # Get time embeddings
            time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
            temb = time_embed[:, -1, :][t_trunc_indices]

            # Discriminator output for generator loss
            fake_output_for_g = self.discriminator(fake_data_at_T_trunc, temb)
            real_labels = torch.ones(fake_output_for_g.size(0), 1).to(self.device)
            loss_gan = F.binary_cross_entropy(fake_output_for_g, real_labels)

            # Add GAN loss term
            loss += loss_gan * self.lambda_gan

            # Compute JS divergence between predicted noise and actual noise
            js_loss = self.js_divergence(predicted[t_trunc_indices], noise[t_trunc_indices])

            # Add JS divergence loss term
            loss += js_loss * self.lambda_js

        return loss

    '''
    discriminator loss function is new
    '''
    def compute_discriminator_loss(self, observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train):
        B, K, L = observed_data.shape

        # Prepare noise and real/fake data at T_trunc
        t = torch.tensor([self.T_trunc - 1] * B).long().to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)

        with torch.no_grad():
            noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
            total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
            predicted = self.diffmodel(total_input, side_info, t)

        # Get real and fake data at T_trunc
        sqrt_alpha_t = torch.sqrt(current_alpha).view(B, 1, 1).expand(B, K, L)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - current_alpha).view(B, 1, 1).expand(B, K, L)
        fake_data_at_T_trunc = (noisy_data - sqrt_one_minus_alpha_t * predicted) / sqrt_alpha_t
        fake_data_at_T_trunc = fake_data_at_T_trunc.detach()
        real_data_at_T_trunc = observed_data

        # Get time embeddings
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        temb = time_embed[:, -1, :]

        # Discriminator output for real and fake samples
        real_output = self.discriminator(real_data_at_T_trunc, temb)
        fake_output = self.discriminator(fake_data_at_T_trunc, temb)

        # Labels for real and fake samples
        real_labels = torch.zeros(B, dtype=torch.long).to(self.device)  # Class 0 for real
        fake_labels = torch.ones(B, dtype=torch.long).to(self.device)   # Class 1 for fake

        # Implement label flipping
        flip_prob = 0.03  # 3% probability to flip labels
        flip_mask_real = torch.rand(B) < flip_prob
        flip_mask_fake = torch.rand(B) < flip_prob

        
        # Flip the labels
        real_labels[flip_mask_real] = 1  # Flip real labels to fake class
        fake_labels[flip_mask_fake] = 0  # Flip fake labels to real class
        
        # Binary Cross-Entropy Loss for discriminator
        D_loss_real = F.cross_entropy(real_output, real_labels)
        D_loss_fake = F.cross_entropy(fake_output, fake_labels)
        D_loss = D_loss_real + D_loss_fake

        return D_loss
    
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else: # if conditional we combine 
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    # impute the missing values
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            z = torch.randn_like(observed_data)  # Random noise as starting point

            # Start reverse process from T_trunc
            current_sample = self.diffmodel(
                self.set_input_to_diffmodel(z, observed_data, cond_mask), side_info, torch.tensor([self.T_trunc - 1]).to(self.device)
            )

            for t in range(self.T_trunc - 1, -1, -1):  # Reverse from T_trunc to 0
                if self.is_unconditional:
                    diff_input = cond_mask * current_sample + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)

                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        if is_train == 1:
            # Compute losses separately during training
            gen_loss = self.compute_generator_loss(observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train)
            discriminator_loss = self.compute_discriminator_loss(observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train)
            return gen_loss, discriminator_loss
        else:
            # Compute generator loss during validation
            gen_loss = self.calc_loss_valid(observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train)
            return gen_loss, None

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch) # get inportant info from batch

        with torch.no_grad(): # dont need gradients durring evaluation
            cond_mask = gt_mask # mask is the groundtruth
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask) # get side info for model

            samples = self.impute(observed_data, cond_mask, side_info, n_samples) # impute the values

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        
        # return everything for later evaluation
        return samples, observed_data, target_mask, observed_mask, observed_tp

# Utility functions like normalization and nonlinearity
def nonlinearity(x):
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# 1D Residual Block with Leaky ReLU and Dropout
'''
Resenet class for the discriminator is new
'''
class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.3, temb_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # Spectral normalization for stability in first convolution layer
        self.conv1 = spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.norm1 = Normalize(in_channels)

        if temb_channels is None:
            raise ValueError("temb_channels (temporal embedding dimension) must be provided!")

        # Temporal embedding projection
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # Second convolution without spectral norm for simplicity
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1))
            else:
                self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x, temb):
        # Main path
        h = self.norm1(x)
        h = self.leaky_relu(h)
        h = self.conv1(h)
        h = h + self.temb_proj(self.leaky_relu(temb)).unsqueeze(-1)
        h = self.leaky_relu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Shortcut path
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

    def forward(self, x, temb):
        # Process x and temb
        h = self.norm1(x)
        h = F.leaky_relu(h, 0.2)
        h = self.conv1(h)  # Spectral norm applied here

        # Apply temporal embedding projection and adjust for batch
        h = h + self.temb_proj(F.leaky_relu(temb, 0.2)).unsqueeze(-1)

        # Use only one normalization, dropout, and simplified second convolution
        h = F.leaky_relu(h, 0.2)
        h = self.dropout(h)
        h = self.conv2(h)  # No spectral norm on the second conv for simplicity

        # Apply shortcut if necessary
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

# 1D Discriminator Block
'''
discriminator class is new
'''
class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_dim=32, temb_dim=64):
        super().__init__()

        # Apply spectral normalization to the first convolutional layer only
        self.input_conv = nn.Sequential(
            spectral_norm(nn.Conv1d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)),  # Spectral norm here
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim)
        )

        self.resblock1 = ResnetBlock1D(hidden_dim, hidden_dim * 2, temb_channels=temb_dim)
        self.resblock2 = ResnetBlock1D(hidden_dim * 2, hidden_dim * 4, temb_channels=temb_dim)

        self.output_layer = nn.Sequential(
            spectral_norm(nn.Conv1d(hidden_dim * 4, 1, kernel_size=3, stride=1, padding=1)),  # No spectral norm here
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x, temb):
        if x.dim() > 3:
            x = x.view(x.size(0), x.size(1), -1)

        x = self.input_conv(x)  # Spectral norm applied here
        x = self.resblock1(x, temb)
        x = self.resblock2(x, temb)
        x = self.output_layer(x)

        return x


# model initialization specific to PM25 data
class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

# model initialization specific to Physio data
class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


# model specific for forecasting
class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        # initialize model configuration
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        # extract specific data from the batch
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    # samples feature to help reduce overfitting
    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind) # shuffle the indices for random extraction
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)

        # return the extracted samples
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask

    # generates the side info for the model to use
    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        # get time embedding
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1) # match shape

        # get feature embedding depending on if all features are used or not
        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)

        # match the right shape
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1) # combined info and mask for conditional

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch) # extract important info
        
        if is_train == 1 and (self.target_dim_base > self.num_sample_features): # use feature samples
            observed_data, observed_mask,feature_id, gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else: # use all features
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0: # if not training then mask is the groundtruth
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id) # get the side info for model

        # calculate loss depending if training for validation
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        # returns appropriate loss
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)


    # evaluate the models forecasting
    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch) # get the important info

        with torch.no_grad(): # no gradients needed
            cond_mask = gt_mask # not training so use groundtruth
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask) # get side info for model to use 

            samples = self.impute(observed_data, cond_mask, side_info, n_samples) # get the imputated values

        return samples, observed_data, target_mask, observed_mask, observed_tp


