import copy
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
import wandb

def train_and_validate_cnnlstm(model, 
                               train_loader, 
                               val_loader, 
                               criterion, 
                               optimizer, 
                               scheduler,  
                               num_epochs, 
                               device):

    model = model.to(device)
    train_losses = []
    val_losses = []

    # log predictions in the first and last epochs
    first_epoch_preds = {"true": [], "pred": []}
    last_epoch_preds = {"true": [], "pred": []}

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for frames, bvp, eda in train_loader:
            frames = frames.to(device)  
            eda = eda.to(device)        

            optimizer.zero_grad()

            outputs = model(frames)     
            outputs = outputs.squeeze(-1)  

            loss = criterion(outputs, eda)  
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)  # accumulate loss weighted by batch size

        # calculate average training loss
        train_loss_avg = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss_avg)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_avg:.4f}")

        # Validation Phase
        model.eval()
        preds = []
        ground_truth = []
        val_loss = 0.0
        with torch.no_grad():
            for frames, bvp, eda in val_loader:
                frames = frames.to(device)
                eda = eda.to(device)

                outputs = model(frames)   

                predictions_list = outputs.cpu().numpy().tolist()
                preds.extend(predictions_list)

                ground_truth_list = eda.cpu().numpy().tolist()
                ground_truth.extend(ground_truth_list)

                outputs = outputs.squeeze(-1)  # (B, T)

                # Validation loss
                loss = criterion(outputs, eda)
                val_loss += loss.item() * frames.size(0)  # Accumulate loss weighted by batch size

                # Predictions for first and last epochs
                if epoch == 0:
                    first_epoch_preds["true"].extend(eda.cpu().numpy().flatten())
                    first_epoch_preds["pred"].extend(outputs.cpu().numpy().flatten())
                if epoch == num_epochs - 1:
                    last_epoch_preds["true"].extend(eda.cpu().numpy().flatten())
                    last_epoch_preds["pred"].extend(outputs.cpu().numpy().flatten())

            # Calculate average validation loss
            val_loss_avg = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss_avg)
            print(f"Validation Loss: {val_loss_avg:.4f}")

        # Step the scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss_avg)  # For schedulers that depend on validation metrics
        else:
            scheduler.step()  # For schedulers that step based on epoch count

        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics to Weights & Biases
        wandb.log({
            "Train Loss": train_loss_avg,
            "Validation Loss": val_loss_avg,
            "Learning Rate": current_lr,
            "Epoch": epoch + 1
        })

        # Optionally, save the best model based on validation loss
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best model updated based on validation loss.")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    wandb.finish()

    return train_losses, val_losses, first_epoch_preds, last_epoch_preds
