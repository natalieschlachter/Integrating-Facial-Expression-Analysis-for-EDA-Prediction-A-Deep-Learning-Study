import torch
import torch.nn as nn

class NegativePearsonCorrelation(nn.Module):
    def __init__(self):
        super(NegativePearsonCorrelation, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Computes the Negative Pearson Correlation between predictions and targets.

        Args:
            y_pred (torch.Tensor): Predicted values. Shape: (batch_size, ...)
            y_true (torch.Tensor): Ground truth values. Shape: (batch_size, ...)

        Returns:
            torch.Tensor: Negative Pearson Correlation loss.
        """
        # Ensure inputs are float tensors
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Flatten the tensors if necessary (e.g., for multi-dimensional outputs)
        if y_pred.dim() > 1:
            y_pred = y_pred.view(y_pred.size(0), -1)
        if y_true.dim() > 1:
            y_true = y_true.view(y_true.size(0), -1)

        # Compute mean for each sample in the batch
        pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
        true_mean = torch.mean(y_true, dim=1, keepdim=True)

        # Compute deviations from mean
        pred_dev = y_pred - pred_mean
        true_dev = y_true - true_mean

        # Compute covariance and variances
        covariance = torch.sum(pred_dev * true_dev, dim=1)
        pred_var = torch.sum(pred_dev ** 2, dim=1)
        true_var = torch.sum(true_dev ** 2, dim=1)

        # Compute Pearson correlation
        pearson_corr = covariance / (torch.sqrt(pred_var) * torch.sqrt(true_var) + 1e-8)

        # Return negative Pearson correlation as loss
        return -pearson_corr.mean()
