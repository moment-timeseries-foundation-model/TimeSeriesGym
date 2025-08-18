import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def generate_synthetic_timeseries(num_samples=1000, t=50, seed=42):
    np.random.seed(seed)
    x = []
    y = []

    for _ in range(num_samples):
        label = np.random.randint(0, 2)  # binary classification (0 or 1)
        y.append(label)

        # Create a synthetic time series based on the label
        if label == 0:  # class 0 has sine pattern
            signal = np.sin(np.linspace(0, 2 * np.pi, t))
        else:  # class 1 has cosine pattern
            signal = np.cos(np.linspace(0, 2 * np.pi, t))

        # Add some random noise
        sample = signal + np.random.normal(0, 0.1, t)
        sample = np.expand_dims(sample, axis=0)
        x.append(sample)

    x = np.stack(x)  # shape: (num_samples, 1, t)
    y = np.array(y)  # shape: (num_samples,)
    return x, y


def prepare(raw: Path, public: Path, private: Path) -> None:
    random_state = 0  # Used for reproducibility
    test_size = 0.1  # 10% of the data will be used for validation
    num_samples = 1000  # Number of samples
    t = 50  # Number of time steps

    # Generate synthetic time series data
    x, y = generate_synthetic_timeseries(num_samples=num_samples, t=t, seed=random_state)
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Save the test ground-truth labels as CSV
    y_test = pd.DataFrame(y_test, columns=["label"])
    y_test.to_csv(private / "test_labels.csv", index=False)

    # Save the test data with dummy labels for testing the predict function
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.zeros(x_test.shape[0], dtype=torch.long)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    torch.save(test_data, public / "test_data.pt")

    # Split the data into training and validation sets
    adjusted_test_size = test_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=adjusted_test_size, random_state=random_state
    )

    # Save a small subset as a tensor for testing the forward function
    x_subset = x_train[:10]
    x_subset = torch.tensor(x_subset, dtype=torch.float32)
    torch.save(x_subset, public / "x.pt")

    # Save the train and validation datasets as PyTorch datasets
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    torch.save(train_data, public / "train_data.pt")

    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    torch.save(val_data, public / "val_data.pt")

    # Download the TensorFlow code files
    os.system(
        f"curl -L -o {public}/resnet.py https://raw.githubusercontent.com/moment-timeseries-foundation-model/dl-4-tsc/resnet-tensorflow/resnet_tensorflow/resnet.py"
    )
    os.system(
        f"curl -L -o {public}/utils.py https://raw.githubusercontent.com/moment-timeseries-foundation-model/dl-4-tsc/resnet-tensorflow/resnet_tensorflow/utils.py"
    )

    # Download the sample submission file
    os.system(
        f"curl -L -o {public}/sample_submission.py https://raw.githubusercontent.com/moment-timeseries-foundation-model/dl-4-tsc/resnet-tensorflow/resnet_pytorch/sample_submission.py"
    )
