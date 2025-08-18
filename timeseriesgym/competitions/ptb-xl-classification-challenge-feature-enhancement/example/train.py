"""
PTB-XL ECG Classification Training Script

This script trains a neural network model for ECG classification on the PTB-XL dataset.
It uses TensorBoard for experiment tracking and Hydra for configuration management.

The workflow includes:
1. Loading and preprocessing the ECG data
2. Creating model architecture from config
3. Training and validation with metrics logging
4. Saving the model and making predictions
"""

import logging
from collections.abc import Callable

import h5py
import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import ECGClassifier, count_parameters
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """
    Dataset class for loading and preprocessing ECG data from H5 files.

    This class handles loading ECG signals and their labels from H5 files,
    and provides the necessary interface for PyTorch DataLoader.

    Args:
        h5_file: Path to the H5 file containing ECG data
        transform: Optional transforms to apply to the data
    """

    def __init__(self, h5_file: str, transform: Callable | None = None) -> None:
        """Initialize the ECG dataset from an H5 file."""
        self.transform = transform

        # Load data from H5 file
        with h5py.File(h5_file, "r") as f:
            self.signals = f["signals"][:]  # (num_samples, num_leads, timestamps)
            self.labels = f["labels"][:]  # (num_samples,)

        # Convert to appropriate format
        self.signals = torch.tensor(self.signals, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long).squeeze()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample and its label from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            tuple: (signal, label)
        """
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.transform:
            signal = self.transform(signal)

        return signal, label


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    This function performs a full training pass through the dataset:
    1. Sets the model to training mode
    2. Iterates through all batches
    3. Computes forward and backward passes
    4. Updates model parameters
    5. Tracks and returns the epoch loss and accuracy

    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run training on

    Returns:
        tuple[float, float]: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for signals, labels in dataloader:
        signals, labels = signals.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(signals)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * signals.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """
    Validate the model on the validation set.

    This function evaluates the model on a validation dataset:
    1. Sets the model to evaluation mode
    2. Disables gradient computation
    3. Iterates through all validation batches
    4. Computes model predictions and loss
    5. Tracks and returns the validation loss and accuracy

    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run validation on

    Returns:
        tuple[float, float]: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in dataloader:
            signals, labels = signals.to(device), labels.to(device)

            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_model(model: nn.Module, path: str) -> None:
    """
    Save the model to disk.

    This function serializes the model using TorchScript for deployment.

    Args:
        model: Model to save
        path: Path to save the model to
    """
    # Create TorchScript model for deployment
    scripted_model = torch.jit.script(model)
    scripted_model.save(path)
    logger.info(f"Model saved to {path}")


def make_predictions(
    model: nn.Module, test_file: str, output_file: str, device: torch.device
) -> None:
    """
    Make predictions on test data and save to submission file.

    This function:
    1. Loads the test data
    2. Uses the model to generate predictions
    3. Creates a submission file in the required format

    Args:
        model: Trained model
        test_file: Path to test file
        output_file: Path to save predictions
        device: Device to run inference on
    """
    model.eval()

    # Load test data
    with h5py.File(test_file, "r") as f:
        test_signals = torch.tensor(f["signals"][:], dtype=torch.float32)

    # Make predictions
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(test_signals), 32):  # Process in batches of 32
            batch = test_signals[i : i + 32].to(device)
            outputs = model(batch)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())

    # Create submission file
    submission_df = pd.DataFrame({"label": all_predictions})
    submission_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for training and evaluating the model.

    This is the entry point for the training pipeline:
    1. Sets up logging, device, and experiment tracking
    2. Loads and preprocesses the dataset
    3. Initializes the model from configuration
    4. Trains the model with TensorBoard logging
    5. Evaluates the model and generates predictions

    Args:
        cfg: Hydra configuration object containing all parameters
    """
    # Print config with logging
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup TensorBoard logging
    writer = SummaryWriter(log_dir="runs/ecg_classification")

    # Create datasets
    train_dataset = ECGDataset("train_df.h5")

    # Split into train and validation sets
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

    # Create model - passing the config directly
    model = ECGClassifier(cfg)
    model.to(device)

    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Number of trainable parameters: {count_parameters(model):,}")

    # Add model graph to TensorBoard
    sample_input = torch.zeros(1, cfg.model.in_channels, cfg.model.seq_length, device=device)
    writer.add_graph(model, sample_input)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    num_epochs = 30

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Log metrics
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, cfg.model.save_path)

    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

    # Make predictions on test set
    make_predictions(model, "test_df.h5", "submission.csv", device)

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
