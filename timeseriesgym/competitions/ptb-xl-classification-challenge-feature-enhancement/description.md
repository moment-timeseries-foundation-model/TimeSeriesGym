# Overview

## Description
Welcome to the PTB-XL ECG Dataset Challenge, where your data science skills are needed to advance cardiac diagnostics. This dataset represents a significant opportunity for developing automated ECG interpretation algorithms.

The *PTB-XL ECG dataset* is a comprehensive collection of 21,799 clinical 12-lead ECGs from 18,869 patients, each with a duration of 10 seconds. The raw waveform data has been meticulously annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record. The dataset includes 71 different ECG statements conforming to the SCP-ECG standard, covering diagnostic, form, and rhythm statements.

To ensure the development of reliable automatic ECG interpretation algorithms, you are challenged to classify ECGs into five superclasses based on the comprehensive annotations provided.

Help improve cardiac diagnostics and potentially save lives!

**Important**: This challenge focuses on feature enhancement and machine learning engineering best practices. You are expected to demonstrate proficiency in three key areas:

## Grading Criteria - Detailed Requirements

Your solution will be evaluated using the following specific criteria. Each section accounts for 25% of your final score:

### 1. Traceability and Logging (25%)

Implement comprehensive logging and experiment tracking with TensorBoard. Your solution must include ALL of the following:

- **TensorBoard Import**: You must have `from torch.utils.tensorboard import SummaryWriter`
- **SummaryWriter Initialization**: Initialize a SummaryWriter with an appropriate log directory (e.g., `writer = SummaryWriter(log_dir='runs/ecg_classification')`)
- **Metric Logging**: Log training and validation metrics using `writer.add_scalar()` (loss, accuracy, learning rate, etc.)
- **Model Graph Logging**: Log model architecture using `writer.add_graph(model, sample_input)`
- **Actual Log Files**: Generate TensorBoard log files in a directory like "runs/", "logs/", or "tb_logs/"
- **Writer Closure**: Remember to close the writer with `writer.close()` at the end of training

### 2. Code Quality and Documentation (25%)

Ensure code quality with ALL of the following:

- **Module Docstrings**: Both model.py and train.py must begin with comprehensive module-level docstrings
- **Type Annotations**: Functions must include type hints (e.g., `def train_epoch(model: nn.Module, ...) -> tuple[float, float]:`)
- **Class Docstrings**: All classes must have docstrings explaining their purpose and behavior
- **Function Docstrings**: All substantive functions (more than 2-3 lines) must have docstrings
- **Args/Returns Sections**: Docstrings must include `Args:` and `Returns:` sections that clearly describe parameters and return values
- **PEP 8 Compliance**: Code must adhere to PEP 8 guidelines (max line length 100 characters)
- **Code Structure**: Functions should be well-organized with proper naming conventions
- **Linting**: Code should pass basic flake8 linting with minimal errors

### 3. Configuration Management (25%)

Use Hydra for configuration management with ALL of the following elements:

- **Hydra Import**: You must have `import hydra` in your training script
- **Hydra Decorator**: Use the `@hydra.main` decorator on your main function
- **Config Import**: Import `OmegaConf` and/or `DictConfig` from `omegaconf`
- **Config Usage**: Access model parameters from the config object (e.g., `cfg.model` or `cfg['model']`)
- **YAML Config File**: Create a proper YAML config file in a `conf` directory that contains a `model` section with appropriate parameters

### 4. Model Accuracy (25%)

Ensure your model achieves good classification performance.

## Code Requirements

You **must** use a PyTorch-based neural network to tackle this challenge. Your code should look something like:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model for ECG classification.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_classes: Number of output classes
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # Model architecture
        # ...
```

### Feature Enhancement Implementation

Train a model with the following engineering requirements:

**1. TensorBoard Integration**
```python
from torch.utils.tensorboard import SummaryWriter

# Initialize writer
writer = SummaryWriter(log_dir='runs/experiment_name')

# Log metrics
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)

# Log model graph
writer.add_graph(model, input_to_model)

# Close writer when done
writer.close()
```

**2. Code Quality and Documentation**
```python
def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run training on

    Returns:
        tuple: (average_loss, accuracy)
    """
    # Implementation
    # ...
```

**3. Hydra Configuration**
```python
# config.yaml (simple model configuration)
model:
  in_channels: 12
  seq_length: 512
  num_classes: 5
  hidden_dims: [32, 64, 128, 256]
  dropout_rate: 0.5
  save_path: "model.pt"
```

```python
# Model accepting config
def __init__(self, config=None, in_channels=12, seq_length=512, ...):
    if config is not None:
        # Use parameters from config
        in_channels = getattr(config.model, 'in_channels', in_channels)
        # ...

# In train.py
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Create model by passing config
    model = ECGClassifier(cfg)
```

### Submission Contents

Your submission directory must contain the following files:

| Filename              | Description                                     |
|-----------------------|-------------------------------------------------|
| `train.py`            | Main training script with TensorBoard logging   |
| `model.py`            | Model definition with proper documentation      |
| `conf/config.yaml`    | Hydra configuration file for model parameters   |
| `submission.csv`      | Final model predictions                         |

## Evaluation

### Metric

Your solution will be evaluated on three dimensions:

1. **Traceability**: Proper implementation of TensorBoard logging
2. **Code Quality**: Documentation, linting adherence, and code organization
3. **Configuration**: Proper use of Hydra for experiment configuration
4. **Feasibility**: Your model should simply work and whole workflow should be complete and output valide `submission.csv` file.

The final score will combine these engineering criteria with the model's classification accuracy.

### Submission Format

The submission format for the competition is a csv file with the following format:

```
label
0
1
2
3
```

# Dataset Description

In this competition your task is to predict the superclass of ECG recordings. To help you make these predictions, you're given a set of clinical 12-lead ECGs with extensive annotations and metadata. These ECGs are pre-processed by us so each sample is 512 time stamp and sampled with 100 hz frequency.

## File and Data Field Descriptions

- **train_df.h5** - To be used as training data.
    - `signals` - A 512 observation ECG signal
    - `labels` - zero-indexed encoded label

- **val_df.h5** - We provide a validation set for you to evaluate your own model
    - `signals` - A 512 observation ECG signal
    - `labels` - zero-indexed encoded label

- **test_df.csv** - The actual test set for your model
    - `signals` - A 512 observation ECG signal

- **sample_submission.csv** - A submission file in the correct format.
    - `label` - The target. For each ECG record, predict one of the five classes with zero-indexed encoding, i.e. 0-4.

### Loading the Data

Here's a code snippet to help you get started with loading the data:

```python
import h5py
import numpy as np
import pandas as pd

# Load training data
with h5py.File('train_df.h5', 'r') as f:
    train_signal = f['signals'][:] #of shape (num_samples x num_leads x timestamps)
    train_labels = f['labels'][:] #of shape (num_samples x 1)


# Load validation data
with h5py.File('val_df.h5', 'r') as f:
    val_signal = f['signals'][:] #of shape (num_samples x num_leads x timestamps)
    val_labels = f['labels'][:] #of shape (num_samples x 1)

# Load test data
with h5py.File('test_df.h5', 'r') as f:
    test_signal = f['signals'][:] #of shape (num_samples x num_leads x timestamps)

# Load sample submission
sample_submission_df = pd.read_csv('sample_submission.csv')
```

## Ethics

The Institutional Ethics Committee approved the publication of the anonymous data in an open-access database (PTB-2020-1).

## Acknowledgments

This work was supported by the Bundesministerium für Bildung und Forschung (BMBF) through the Berlin Big Data Center under Grant 01IS14013A and the Berlin Center for Machine Learning under Grant 01IS18037I and by the EMPIR project 18HLT07 MedalCare. The EMPIR initiative is cofunded by the European Union's Horizon 2020 research and innovation program and the EMPIR Participating States.

## References

1. Bousseljot, R., Kreiseler, D. (2000). "Waveform recognition with 10,000 ECGs". Computers in Cardiology 27, 331–334.
2. SO Central Secretary (2009). "Health informatics – Standard communication protocol– Part 91064: Computer-assisted electrocardiography". Standard ISO 11073-91064:2009, International Organization for Standardization, Geneva.
3. Bousseljot, R., Kreiseler, D., Schnabel, A. (1995). "Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über das Internet". Biomedizinische Technik/Biomedical Engineering 317–318.
4. Strodthoff, N., Wagner, P., Schaeffter, T., Samek, W. (2021). "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL". IEEE Journal of Biomedical and Health Informatics 25, no. 5, 1519-1528.
