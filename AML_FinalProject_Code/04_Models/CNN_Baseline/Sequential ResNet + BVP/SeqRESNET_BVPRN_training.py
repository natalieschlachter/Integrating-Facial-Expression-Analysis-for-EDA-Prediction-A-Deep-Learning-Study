import torch
import wandb


def train_and_validate_windowed_ubfc(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device
):

    model.to(device)
    train_losses = []
    val_losses = []

    
    first_epoch_preds = {"true": [], "pred": []}
    last_epoch_preds = {"true": [], "pred": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for frames, bvp_values, eda_values in train_loader:
        
        
            frames = frames.to(device)       
            bvp_values = bvp_values.to(device) 
            eda_values = eda_values.to(device) 

        
            optimizer.zero_grad()

        
            outputs = model(frames, bvp_values)  
            outputs = outputs.squeeze(-1)        

            loss = criterion(outputs, eda_values)  
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        train_loss_avg = running_loss / len(train_loader)
        train_losses.append(train_loss_avg)
        print(f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {train_loss_avg:.4f}")

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, bvp_values, eda_values in val_loader:
                frames = frames.to(device)        
                bvp_values = bvp_values.to(device)
                eda_values = eda_values.to(device) 

                outputs = model(frames, bvp_values)
                outputs = outputs.squeeze(-1)       

                loss = criterion(outputs, eda_values)
                val_loss += loss.item()

                # predictions for first and last epochs
                if epoch == 0:  # first epoch
                    first_epoch_preds["true"].extend(eda_values.cpu().numpy().flatten())
                    first_epoch_preds["pred"].extend(outputs.cpu().numpy().flatten())
                if epoch == num_epochs - 1:  # last epoch
                    last_epoch_preds["true"].extend(eda_values.cpu().numpy().flatten())
                    last_epoch_preds["pred"].extend(outputs.cpu().numpy().flatten())

            val_loss_avg = val_loss / len(val_loader)
            val_losses.append(val_loss_avg)
            print(f"           Validation Loss: {val_loss_avg:.4f}")

            wandb.log({
            "Train Loss": train_loss_avg,
            "Validation Loss": val_loss_avg,
            "Epoch": epoch + 1
            })

    wandb.finish()

    return train_losses, val_losses, first_epoch_preds, last_epoch_preds
