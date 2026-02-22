import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

def test_model(model, test_loader, criterion, device='cuda', num_samples=32):
    """
    Test the model on the test set.
    """
    # Set the model to evaluation mode
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to track gradients during testing
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    wandb.log({"test loss": avg_test_loss})
    data = [[i, float(t), float(p)] for i, (t, p) in enumerate(zip(all_labels, all_preds))]
    table = wandb.Table(data=data, columns=["index", "actual", "prediction"])
    wandb.log({"test_predictions_table": table})

    return avg_test_loss, all_preds, all_labels
