# Overview

## Description
Welcome to the PTB-XL ECG Dataset Challenge, where your data science skills are needed to advance cardiac diagnostics. This dataset represents a significant opportunity for developing automated ECG interpretation algorithms.

The *PTB-XL ECG dataset* is a comprehensive collection of 21,799 clinical 12-lead ECGs from 18,869 patients, each with a duration of 10 seconds. The raw waveform data has been meticulously annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record. The dataset includes 71 different ECG statements conforming to the SCP-ECG standard, covering diagnostic, form, and rhythm statements.

To ensure the development of reliable automatic ECG interpretation algorithms, you are challenged to classify ECGs into five superclasses based on the comprehensive annotations provided.

Help improve cardiac diagnostics and potentially save lives!

**Important**: This challenge requires you to perform an explicit hyperparameter search. Since you will be evaluated on your optimization process, please document every step in your solution file.

## Code Requirements

You **must** use a PyTorch-based neural network to tackle this challenge. Your code should look something like:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    …
```

### Hyperparameter Tuning

Train two separate networks:

1. **Baseline model**
2. **Improved model**

After training, serialize each with TorchScript:

```python
# Save baseline
baseline_scripted = torch.jit.script(baseline_model)
baseline_scripted.save("baseline_model.pt")

# Save improved
improved_scripted = torch.jit.script(improved_model)
improved_scripted.save("model.pt")
```

### Submission Contents

Your submission directory must contain exactly these four files:

| Filename                   | Description                            |
|----------------------------|----------------------------------------|
| `baseline_submission.csv`  | Predictions from your baseline model   |
| `submission.csv`           | Predictions from your improved model   |
| `baseline_model.pt`        | TorchScript file for baseline model    |
| `model.pt`                 | TorchScript file for improved model    |

> **Important:** You are required to conduct hyperparameter optimization, or the submission will result in an **invalid submission**.

## Evaluation

### Metric

Submissions are evaluated based on their classification accuracy, the percentage of predicted labels that are correct.

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
