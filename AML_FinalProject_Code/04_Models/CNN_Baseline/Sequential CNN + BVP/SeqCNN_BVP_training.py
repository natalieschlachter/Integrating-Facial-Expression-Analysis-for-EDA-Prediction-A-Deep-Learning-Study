import copy
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
import wandb

def train_and_validate_cnnlstm_bvp(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler,
    num_epochs, 
    device,
    patience  
):
  
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for sequences, bvp_values, eda_targets in train_loader:
            
            sequences = sequences.to(device)        
            bvp_values = bvp_values.to(device)      
            eda_targets = eda_targets.to(device)    

            optimizer.zero_grad()

            # forward 
            outputs = model(sequences, bvp_values)  
            outputs = outputs.squeeze(-1)           

            #  loss
            loss = criterion(outputs, eda_targets)  
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)

        
        train_loss_avg = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss_avg)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_avg:.4f}")

        # val 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, bvp_values, eda_targets in val_loader:
              
                sequences = sequences.to(device)
                bvp_values = bvp_values.to(device)
                eda_targets = eda_targets.to(device)

                # forward
                outputs = model(sequences, bvp_values)  
                outputs = outputs.squeeze(-1)           

                # loss
                loss = criterion(outputs, eda_targets)
                val_loss += loss.item() * sequences.size(0)

        
        val_loss_avg = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss_avg)
        print(f"Validation Loss: {val_loss_avg:.4f}")
        
        # step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss_avg)  
        else:
            scheduler.step() 

        
        current_lr = optimizer.param_groups[0]['lr']

        
        wandb.log({
            "Train Loss": train_loss_avg,
            "Validation Loss": val_loss_avg,
            "Learning Rate": current_lr,
            "Epoch": epoch + 1
        })

        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"Validation loss decreased. Saving best model.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    
    model.load_state_dict(best_model_wts)
    wandb.finish()
    return train_losses, val_losses, model
