"""
PTB-XL ECG Classification Model

This module defines a PyTorch model for ECG classification on the PTB-XL dataset.
It implements a convolutional neural network for 12-lead ECG signal classification.
"""

import torch
import torch.nn as nn


class ECGClassifier(nn.Module):
    """
    Neural network model for ECG signal classification.

    This model processes 12-lead ECG data and classifies it into one of five diagnostic
    superclasses. The architecture uses 1D convolutional layers followed by batch
    normalization and max pooling.

    Args:
        config: Configuration object or dictionary with model parameters
        in_channels: Number of input channels (ECG leads)
        seq_length: Sequence length of the ECG signal
        num_classes: Number of output classes
        hidden_dims: List of hidden dimension sizes for convolutional layers
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        config: object | None = None,
        in_channels: int = 12,
        seq_length: int = 512,
        num_classes: int = 5,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.5,
    ) -> None:
        """Initialize the ECGClassifier model with given parameters."""
        super().__init__()

        # Default value for hidden_dims if None
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # If config is provided, use its parameters
        if config is not None:
            if hasattr(config, "model"):
                # Hydra config case
                config = config.model

            in_channels = getattr(config, "in_channels", in_channels)
            seq_length = getattr(config, "seq_length", seq_length)
            num_classes = getattr(config, "num_classes", num_classes)
            hidden_dims = getattr(config, "hidden_dims", hidden_dims)
            dropout_rate = getattr(config, "dropout_rate", dropout_rate)

        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_classes = num_classes

        # Feature extraction layers
        self.conv_layers = nn.ModuleList()

        # First convolutional layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dims[0], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
        )

        # Additional convolutional layers
        for i in range(len(hidden_dims) - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dims[i], hidden_dims[i + 1], kernel_size=5, stride=1, padding=2
                    ),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
            )

        # Calculate the size of flattened features after convolutions
        self._calculate_feature_size()

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def _calculate_feature_size(self) -> None:
        """
        Calculate the flattened feature size after convolutions.

        This method creates a dummy input tensor and passes it through the
        convolutional layers to determine the flattened feature size for
        the linear layers.
        """
        # Create a dummy input to find output size
        x = torch.zeros(1, self.in_channels, self.seq_length)

        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Get size
        self.feature_size = x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_length)

        Returns:
            torch.Tensor: Logits for each class
        """
        # Ensure the input has the right shape
        if x.dim() == 3:
            # If the input is (batch_size, in_channels, seq_length)
            pass
        elif x.dim() == 4:
            # If the input is (batch_size, in_channels, seq_length, 1)
            x = x.squeeze(-1)

        # Feature extraction
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the features
        x = x.view(x.size(0), -1)

        # Classification
        logits = self.classifier(x)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_length)

        Returns:
            torch.Tensor: Predicted class indices
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


# Additional utility functions for model analysis
def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: tuple[int, ...]) -> str:
    """
    Generate a text summary of the model architecture.

    Args:
        model: PyTorch model
        input_size: Input tensor shape (without batch dimension)

    Returns:
        str: Model summary string
    """
    import sys
    from io import StringIO

    # Create a string buffer to capture the print output
    buffer = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    # Print model summary
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)
    print(f"Input shape: {input_size}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("=" * 80)
    print(model)

    # Restore stdout
    sys.stdout = old_stdout

    return buffer.getvalue()
