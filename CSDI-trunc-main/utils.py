import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import torch

torch.autograd.set_detect_anomaly(True)

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername="",
):
    if foldername != "":
        output_path = foldername + "/model.pth"

    # Separate optimizers
    generator_params = [param for name, param in model.named_parameters() if 'discriminator' not in name]
    discriminator_params = [param for name, param in model.named_parameters() if 'discriminator' in name]

    # Create separate optimizers
    optimizer_G = Adam(generator_params, lr=config["lr"], weight_decay=1e-6)
    optimizer_D = Adam(discriminator_params, lr=config["lr"], weight_decay=1e-6)

    # Learning rate schedulers
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_G, milestones=[p1, p2], gamma=0.1
    )

    lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_D, gamma=0.99  # Reduce by 1% each epoch
    )

    best_valid_loss = float('inf')  # Initialize best validation loss
    warmup_epochs = 20  # Number of epochs to warm up the generator
    for epoch_no in range(config["epochs"]):
        avg_gen_loss = 0
        avg_disc_loss = 0
        discriminator_updates = 0  # Initialize counter
        n_critic = 12  # Update discriminator every n_critic iterations

        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                # Prepare data
                (
                    observed_data,
                    observed_mask,
                    observed_tp,
                    gt_mask,
                    for_pattern_mask,
                    _,
                ) = model.process_data(train_batch)

                if model.target_strategy != "random":
                    cond_mask = model.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
                else:
                    cond_mask = model.get_randmask(observed_mask)
                side_info = model.get_side_info(observed_tp, cond_mask)

                # ----------------------
                # Train Generator
                # ----------------------
                optimizer_G.zero_grad()
                gen_loss = model.compute_generator_loss(
                    observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train=1
                )
                gen_loss.backward()
                optimizer_G.step()
                avg_gen_loss += gen_loss.item()

                # ----------------------
                # Train Discriminator (after warmup)
                # ----------------------
                if epoch_no >= warmup_epochs and batch_no % n_critic == 0:
                    optimizer_D.zero_grad()
                    discriminator_loss = model.compute_discriminator_loss(
                        observed_data, cond_mask, observed_mask, side_info, observed_tp, is_train=1
                    )
                    discriminator_loss.backward()
                    optimizer_D.step()
                    avg_disc_loss += discriminator_loss.item()
                    discriminator_updates += 1  # Increment counter

                # Update progress
                it.set_postfix(
                    ordered_dict={
                        "avg_gen_loss": avg_gen_loss / batch_no,
                        "avg_disc_loss": (avg_disc_loss / discriminator_updates) if discriminator_updates > 0 else 0,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            # Step the discriminator scheduler every epoch
            if epoch_no >= warmup_epochs:
                lr_scheduler_D.step()

        # Validation loop
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        gen_loss, _ = model(valid_batch, is_train=0)
                        avg_loss_valid += gen_loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            avg_loss_valid /= len(valid_loader)

            # Use the validation loss to update the generator's learning rate scheduler
            lr_scheduler_G.step(avg_loss_valid)

            if avg_loss_valid < best_valid_loss:
                best_valid_loss = avg_loss_valid

                # Display the best loss of the epoch
                print(
                    f"\nBest validation loss updated to {avg_loss_valid} at epoch {epoch_no}"
                )

    # Save only the final model at the end of training
    if foldername != "":
        torch.save(model.state_dict(), output_path)

# get the quantile loss for CRPS using weighted error based on q
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

# get the denominator for CRPS
def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

# get the actual CRPS
def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler # ground truth values
    forecast = forecast * scaler + mean_scaler # model predicted values

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)): # loop through each quantile
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points) # get the loss at each quantile
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1) # the ground truth values
    forecast = forecast * scaler + mean_scaler # the model predicted values

    quantiles = np.arange(0.05, 1.0, 0.05) 
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)): # for each quantile get the quantile loss
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom # add it all up to get the CRPS score
    return CRPS.item() / len(quantiles) # average it

# evaluate the models performance 
def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad(): # we don't need to calculate the gradients
        model.eval() # set the model into evaluation mode

        # set up all scores to default 0
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        # initialize accumulators for storing results
        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        # iterate through all the test batches
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                
                # get outputs from evaluation
                samples, c_target, eval_points, observed_points, observed_time = output 

                # permute the tensors to match the correct dimensions for processing
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                # add ouputs to the list of results
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # calculate MSE and MAE for current batch
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                # update the overall MSE and MAE
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            # save general results
            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump( # this part actually does the saving
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            # get CRPS and CRPS sum
            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump( # save CRPS and CRPS sum
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )

                # display all the results from the evaluation
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)
