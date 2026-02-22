import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def test_cnnlstm(model, 
                test_loader, 
                criterion, 
                device, 
                log_predictions=False):

    model.eval()
    test_loss = 0.0
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for batch_idx, (frames, bvp, eda) in enumerate(test_loader):
            
            frames = frames.to(device)  
            eda = eda.to(device)        

    
            outputs = model(frames)     
            outputs = outputs.squeeze(-1)  

    
            loss = criterion(outputs, eda)
            test_loss += loss.item()

    
            predictions.extend(outputs.cpu().numpy().flatten())
            ground_truth.extend(eda.cpu().numpy().flatten())

    
    test_loss_avg = test_loss / len(test_loader.dataset)

    predictions_np = np.array(predictions)
    ground_truth_np = np.array(ground_truth)
    mae = np.mean(np.abs(predictions_np - ground_truth_np))
    rmse = np.sqrt(np.mean((predictions_np - ground_truth_np) ** 2))

    test_metrics = {
        "Test Loss": test_loss_avg,
        "MAE": mae,
        "RMSE": rmse
    }

    print(f"Test Loss: {test_loss_avg:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return test_loss_avg, test_metrics, predictions, ground_truth

