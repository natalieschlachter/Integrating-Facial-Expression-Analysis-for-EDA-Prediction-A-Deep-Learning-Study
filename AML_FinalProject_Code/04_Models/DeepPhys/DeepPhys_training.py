import os
import random
import shutil
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

class NegativePearsonCorrelation(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(NegativePearsonCorrelation, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Computes Negative Pearson Correlation for 4D input and 2D target tensors
        """
        try:
            if y_pred.dim() != y_true.dim():
                y_pred = y_pred.view(y_pred.size(0), -1)  # Flatten to (batch_size, features)
                y_pred = y_pred.mean(dim=1, keepdim=True)  # Average to (batch_size, 1)

            y_pred = y_pred.float()
            y_true = y_true.float()
            
            # Normalize predictions to [0,1] range like targets
            y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + self.epsilon) 
            
            # Compute batch statistics
            pred_mean = y_pred.mean(dim=0)
            true_mean = y_true.mean(dim=0)
            pred_diff = y_pred - pred_mean
            true_diff = y_true - true_mean
            
            # Compute correlation
            numerator = (pred_diff * true_diff).sum(dim=0)
            denominator = torch.sqrt(
                (pred_diff ** 2).sum(dim=0) * (true_diff ** 2).sum(dim=0) + self.epsilon
            )
            
            correlation = numerator / denominator
            correlation = torch.clamp(correlation, min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
            
            return -correlation.mean()

        except Exception as e:
            print(f"Error in NPC loss: {str(e)}")
            raise e

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5, model_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.model_path = model_path
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.model_path)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs=20, device='cuda', early_stopping=None):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Create the loss curves plot for W&B
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    
    for epoch in range(epochs):
        ######### Training Phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_ground_truth = []
        print(f"Epoch {epoch + 1}/{epochs}")
        train_progress = tqdm(train_loader, desc="Training")

        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
            
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_ground_truth.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        ########### Validation Phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_ground_truth = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_ground_truth.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss) # Learning rate scheduling

        # loss curves plot
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        ax_loss.plot(range(1, epoch + 2), train_losses, label='Training Loss', marker='o')
        ax_loss.plot(range(1, epoch + 2), val_losses, label='Validation Loss', marker='o')
        ax_loss.set_title('Training and Validation Loss Curves')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # prediction vs actual 
        fig_pred, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        ax1.plot(train_predictions[:100], label='Predictions', alpha=0.7)
        ax1.plot(train_ground_truth[:100], label='Ground Truth', alpha=0.7)
        ax1.set_title('Training: Predictions vs Ground Truth')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(val_predictions[:100], label='Predictions', alpha=0.7)
        ax2.plot(val_ground_truth[:100], label='Ground Truth', alpha=0.7)
        ax2.set_title('Validation: Predictions vs Ground Truth')
        ax2.legend()
        ax2.grid(True)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "loss_curves": wandb.Image(fig_loss),
            "predictions_vs_ground_truth": wandb.Image(fig_pred)
        })

        plt.close(fig_loss)
        plt.close(fig_pred)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if early_stopping is not None: # Early stopping check
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    wandb.run.summary.update({
        "best_val_loss": min(val_losses),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_epochs": len(train_losses),
        "early_stopped": early_stopping.early_stop if early_stopping else False,
    })

    return train_losses, val_losses


def setup_training(model, learning_rate=0.0001, weight_decay=0.0001):
    criterion = NegativePearsonCorrelation() #nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
    
    return criterion, optimizer, scheduler, early_stopping


def plot_loss_curves(train_losses, val_losses):
    """
    Plots the training and validation loss curves on the same graph.

    Args:
        train_losses (list or array-like): List of training loss values per epoch.
        val_losses (list or array-like): List of validation loss values per epoch.
    """
    epochs = range(1, len(train_losses) + 1)  # Create a range of epochs

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(epochs)  # Ensure all epochs are displayed on the x-axis
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


