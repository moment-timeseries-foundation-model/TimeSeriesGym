# Overview

## Description
Welcome to the PTB-XL ECG Dataset Challenge, where your data science skills are needed to advance cardiac diagnostics. This dataset represents a significant opportunity for developing automated ECG interpretation algorithms.

The *PTB-XL ECG dataset* is a comprehensive collection of 21,799 clinical 12-lead ECGs from 18,869 patients, each with a duration of 10 seconds. The raw waveform data has been meticulously annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record. The dataset includes 71 different ECG statements conforming to the SCP-ECG standard, covering diagnostic, form, and rhythm statements.

To ensure the development of reliable automatic ECG interpretation algorithms, you are challenged to classify ECGs into five superclasses based on the comprehensive annotations provided.

Help improve cardiac diagnostics and potentially save lives!

**Important**: This challenge focuses on feature enhancement and machine learning engineering best practices. You are expected to demonstrate proficiency in three key areas:

## Grading Criteria - Detailed Requirements

Your score is the accuracy of your `submission.csv` on the test set, provided that your solution passes **every** check of criteria 1-3 below; if any check fails, your score is 0. Each check is pass/fail. The checks read `train.py`, `model.py` and `conf/config.yaml` (comments are ignored, and your code is not run), and look for TensorBoard event files in your submission folder.

### 1. Traceability and Logging

Implement logging and experiment tracking with TensorBoard. `train.py` must contain ALL of the following:

- **TensorBoard Import**: `from torch.utils.tensorboard import SummaryWriter` (or `import torch.utils.tensorboard`)
- **SummaryWriter Initialization**: a `SummaryWriter(...)` call with an appropriate log directory (e.g., `writer = SummaryWriter(log_dir='runs/ecg_classification')`)
- **Metric Logging**: `writer.add_scalar(...)` calls that log training and validation metrics (loss, accuracy, learning rate, etc.)
- **Model Graph Logging**: a `writer.add_graph(model, sample_input)` call that logs the model architecture

And your submission folder must contain:

- **Actual Log Files**: at least one TensorBoard event file (`events.out.tfevents.*`, as written by `SummaryWriter`) under `runs/`, `logs/` or `tb_logs/`. Run your training so that these files are written, and keep them.

Close the writer with `writer.close()` at the end of training, so that the logs are complete.

### 2. Code Quality and Documentation

`model.py` and `train.py` must pass ALL of the following:

- **Valid Python**: both files parse as Python
- **Module Docstrings**: both files begin with a module-level docstring
- **Type Annotations**: in each file, every function and method annotates all of its parameters (except `self` and `cls`) and its return type (e.g., `def train_epoch(model: nn.Module, ...) -> tuple[float, float]:`); `__init__` may leave out `-> None`. Each file defines at least one function.
- **Class Docstrings**: `model.py` defines at least one class, and every class in it has a docstring
- **Function Docstrings**: in each file, every function with more than 3 statements directly in its body has a docstring, and at least one documented function has 3 or more (counting the docstring)
- **Args/Returns Sections**: `Args:` and `Returns:` sections in your docstrings (each must appear in at least one docstring of the two files; describe the parameters and return values of every function, as in the example below)
- **Linting (PEP 8)**: `flake8 --isolated --disable-noqa --extend-ignore=E203 --max-line-length=100` reports no problems in either file. flake8's default checks apply, except E203 (the whitespace that black and ruff put around a slice's colon, as in `x[i : i + 1]`); `# noqa` comments and flake8 configuration files are ignored.

### 3. Configuration Management

Use Hydra for configuration management. `train.py` must contain ALL of the following:

- **Hydra Import**: `import hydra` (or `from hydra import ...`)
- **Hydra Decorator**: the `@hydra.main` decorator on your main function
- **Config Import**: `OmegaConf` and/or `DictConfig`, imported from `omegaconf`
- **Config Usage**: the model is built from the config's `model` section: `train.py` passes the section (`cfg.model`, `cfg['model']`) or a value read from it (e.g., `cfg.model.hidden_dims`) to a class or function of `model.py` or to `hydra.utils.instantiate`, or it passes the config to a class or function of `model.py` that reads the `model` section of its parameter (as `ECGClassifier(cfg)` in the example below, whose `__init__` reads `config.model`). Printing `cfg.model`, or only mentioning it, does not count.

And your solution must contain:

- **YAML Config File**: `conf/config.yaml`, with a `model` section that holds the model's parameters

### 4. Model Accuracy

If every check above passes, your score is the classification accuracy of `submission.csv`: the fraction of test ECGs whose label you predict correctly. Otherwise your score is 0. The grader also reports a sub-score for each of criteria 1-3 (the weighted share of its checks that pass, with partial credit for fewer than 20 flake8 problems); the sub-scores do not change your score.

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
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
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
```yaml
# config.yaml (simple model configuration)
model:
  in_channels: 12
  seq_length: 1000
  num_classes: 5
  hidden_dims: [32, 64, 128, 256]
  dropout_rate: 0.5
  save_path: "model.pt"
```

```python
# Model accepting config
def __init__(
    self, config: DictConfig | None = None, in_channels: int = 12, seq_length: int = 1000
):
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

Your score is the classification accuracy of `submission.csv` if your solution passes every check of the Grading Criteria above, and 0 otherwise:

1. **Traceability**: TensorBoard logging, with its event files in your submission
2. **Code Quality**: Documentation, type annotations, and no flake8 problems
3. **Configuration**: Hydra configuration with a `conf/config.yaml`
4. **Feasibility**: Your model should simply work: the whole workflow must run and output a valid `submission.csv` file, whose accuracy is your score.

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

In this competition your task is to predict the superclass of ECG recordings. To help you make these predictions, you're given a set of clinical 12-lead ECGs with extensive annotations and metadata. These ECGs are pre-processed by us so each sample is a 12-lead recording of 1000 time steps (10 seconds sampled at 100 Hz).

## File and Data Field Descriptions

- **train_df.h5** - To be used as training data.
    - `signals` - A 12-lead ECG signal of 1000 observations (shape 12 x 1000)
    - `labels` - zero-indexed encoded label

- **val_df.h5** - We provide a validation set for you to evaluate your own model
    - `signals` - A 12-lead ECG signal of 1000 observations (shape 12 x 1000)
    - `labels` - zero-indexed encoded label

- **test_df.h5** - The actual test set for your model
    - `signals` - A 12-lead ECG signal of 1000 observations (shape 12 x 1000)

- **sample_submission.csv** - A submission file in the correct format, with one row per ECG in `test_df.h5`, in the same order (rows are matched by position).
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
    train_labels = f['labels'][:] #of shape (num_samples,)


# Load validation data
with h5py.File('val_df.h5', 'r') as f:
    val_signal = f['signals'][:] #of shape (num_samples x num_leads x timestamps)
    val_labels = f['labels'][:] #of shape (num_samples,)

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
