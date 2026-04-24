import os
import numpy as np
import torch
import torch.nn as nn


# Define RMSE loss as a custom function
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE) Loss.
    """
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-7
        percentage_errors = torch.abs((y_true - y_pred) / (y_true + epsilon))
        return torch.mean(percentage_errors) * 100


# Map loss function strings to PyTorch loss functions
LOSS_FUNCTIONS_MAP = {
    'mse': nn.MSELoss(),
    'mae': nn.L1Loss(),
    # 'huber': nn.HuberLoss(),
    # 'cross_entropy': nn.CrossEntropyLoss(),
    'rmse': RMSELoss(),
    'mape': MAPELoss(),
}


def train_one_epoch(
        model,
        train_loader,
        optimizer,
        train_losses,
        loss_func='mse',
        verbose=True
):
    """
    Performs training for one epoch.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for gradient updates.
        train_losses (list): List to store training losses across epochs.
        loss_func (str): Loss function to use (default: 'mse').
        verbose (bool): Whether to print training loss (default: True).

    Returns:
        None
    """
    model.train()  # Set the model to training mode
    criterion = LOSS_FUNCTIONS_MAP.get(loss_func.lower(), nn.MSELoss())  # Default to MSELoss
    epoch_loss = []

    for data, target in train_loader:
        optimizer.zero_grad()  # Reset gradients
        output = model(data)  # Forward pass
        loss = criterion(target.squeeze(), output.squeeze())  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        epoch_loss.append(loss.item())

    avg_loss = sum(epoch_loss) / len(train_loader)
    if verbose:
        print(f"Training set: Avg. loss: {avg_loss:.3f}")
    train_losses.append(avg_loss)


def validation(
        model,
        vali_loader,
        vali_losses=None,
        target_scaler=None,
        loss_func='mse',
        verbose=True,
        testing=False
):
    """
    Performs validation for one epoch.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        vali_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        vali_losses (list): List to store validation losses across epochs (optional).
        target_scaler (object): Scaler for inverse-transforming targets (optional).
        loss_func (str): Loss function to use (default: 'mse').
        verbose (bool): Whether to print validation loss (default: True).
        testing (bool): Whether to return predictions and targets (default: False).

    Returns:
        float: Average validation loss, or predictions and targets if `testing=True`.
    """
    model.eval()  # Set the model to evaluation mode
    criterion = LOSS_FUNCTIONS_MAP.get(loss_func.lower(), nn.MSELoss())  # Default to MSELoss
    epoch_loss = []
    predictions = []
    actuals = []

    with torch.no_grad():  # Disable gradient computation for validation
        for data, target in vali_loader:
            output = model(data)  # Forward pass
            if target_scaler:
                # Apply inverse transform if scaler is provided
                target = torch.tensor(
                    target_scaler.inverse_transform(target.reshape(-1, 1).cpu().numpy())
                ).view(-1)  # Flatten to shape [8748]
                output = torch.tensor(
                    target_scaler.inverse_transform(output.cpu().numpy())
                ).view(-1)  # Flatten to shape [8748]

                predictions.append(output)
                actuals.append(target)

            loss = criterion(target, output.squeeze())  # Compute loss
            epoch_loss.append(loss.item())

    avg_loss = sum(epoch_loss) / len(vali_loader)
    if verbose:
        print(f"Validation set: Avg. loss: {avg_loss:.3f}")
    if vali_losses is not None:
        vali_losses.append(avg_loss)

    if testing:
        return predictions, actuals
    else:
        return avg_loss


def train_model(
        model,
        optimizer,
        scheduler,
        train_loader,
        vali_loader,
        target_scaler=None,
        n_epochs=50,
        early_stopping_threshold=5,
        loss_func='mse',
        save_path=None,
        verbose=True
):
    """
    Main training loop for the model.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for gradient updates.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        vali_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        target_scaler (object): Scaler for inverse-transforming targets (optional).
        n_epochs (int): Number of epochs to train (default: 50).
        early_stopping_threshold (int): Early stopping criterion (default: 5 epochs).
        loss_func (str): Loss function to use (default: 'mse').
        save_path (str): Path to save the best model parameters (optional).
        verbose (bool): Whether to print progress during training (default: True).

    Returns:
        tuple: Training losses, validation losses, validation counter (epochs).
    """
    train_losses = []
    vali_losses = []
    train_counter = list(range(1, n_epochs + 1))
    vali_counter = list(range(1, n_epochs + 1))
    best_valid_loss = float('inf')
    patience = 0  # Counter for early stopping

    # Initial validation before training
    validation(model, vali_loader, vali_losses, target_scaler, loss_func, verbose)

    for epoch in range(1, n_epochs + 1):
        if verbose:
            print(f"\nEpoch: {epoch}")

            # Training and validation for the current epoch
        train_one_epoch(model, train_loader, optimizer, train_losses, loss_func, verbose)
        valid_loss = validation(model, vali_loader, vali_losses, target_scaler, loss_func, verbose)

        # Save model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience = 0
            if save_path:
                torch.save(model.state_dict(), os.path.join(save_path, 'NN_params.pt'))
        else:
            patience += 1

            # Check early stopping
        if patience >= early_stopping_threshold:
            if verbose:
                print("Early stopping triggered.")
            train_counter = train_counter[:epoch]
            vali_counter = vali_counter[:epoch + 1]
            break

        # Step the scheduler
        scheduler.step(valid_loss)

        # Print best losses
    if verbose:
        best_train_epoch = train_losses.index(min(train_losses))
        best_vali_epoch = vali_losses.index(min(vali_losses))
        print(f"\nBest training loss: {min(train_losses):.3f} (Epoch: {best_train_epoch + 1})")
        print(f"Best validation loss: {min(vali_losses):.3f} (Epoch: {best_vali_epoch + 1})")

    return train_losses, vali_losses, vali_counter
