import torch

def test_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    
    true_eda = []
    pred_eda = []
    
    with torch.no_grad():
        for sequences, eda_targets, bvp_values in test_loader:
            # Move data to device
            sequences = sequences.to(device)
            eda_targets = eda_targets.to(device)
            bvp_values = bvp_values.to(device)

            # Forward pass
            outputs = model(sequences, bvp_values)  # => (B, T, 1)
            outputs = outputs.squeeze(-1)          # => (B, T)

            # If your dataset only has a single EDA per sequence, replicate it.
            if len(eda_targets.shape) == 1:
                eda_targets = eda_targets.unsqueeze(-1).repeat(1, outputs.size(1))

            # Compute loss
            loss = criterion(outputs, eda_targets)
            test_loss += loss.item()

            # Store predictions and targets for later analysis
            true_eda.extend(eda_targets.cpu().numpy().flatten())
            pred_eda.extend(outputs.cpu().numpy().flatten())

    # Average test loss
    test_loss /= len(test_loader)

    return test_loss, true_eda, pred_eda