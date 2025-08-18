# UCR Anomaly Detection

## Overview
This competition focuses on univariate time series anomaly detection using the UCR Time Series Anomaly Datasets. The collection includes 250 distinct datasets spanning various domains, each presenting a unique anomaly detection challenge.

**IMPORTANT:**
You have to solve datasets one by one, as they are provided in `datasets.txt` file. Each line is a dataset name. You can pass this name to the `UCRAnomalyDataset` class to load the dataset.

## Challenge Description
Anomaly detection in time series is a critical task in many fields including manufacturing, healthcare, finance, and infrastructure maintenance. Your goal is to identify anomalous segments in time series data by analyzing patterns and detecting deviations from normal behavior.

## Dataset
Each dataset consists of:
- A training set containing only normal patterns
- A test set containing one anomalous segment

The datasets vary in length, sampling frequency, and the nature of the anomalies they contain. Datasets that you need to process are specified in dataset_names.txt, each line is a dataset name. You can pass this name to the `UCRAnomalyDataset` class to load the dataset.

## Evaluation
Submissions are evaluated using the adjusted F1 score, which balances precision and recall in anomaly detection. The scoring function finds the optimal threshold of the score that maximizes the F1 score. When the optimal threshold is found, the best F1 score is used as the score for that dataset.

### Adjusted F1 Score
The adjusted F1 score addresses a common challenge in time series anomaly detection: the exact boundaries of anomalies are often ambiguous. The evaluation metric works as follows:

1. For each dataset, your model provides an anomaly score for each data point (higher values indicate higher likelihood of anomaly)
2. The evaluation divides the range of scores into multiple thresholds
3. At each threshold, predictions are converted to binary (anomaly/not anomaly)
4. Before calculating the F1 score, predictions are adjusted to account for slight misalignments in anomaly boundaries
5. The F1 score is calculated for each threshold, and the maximum F1 score is used as the final score for that dataset

This adjustment ensures that your model is rewarded for correctly identifying anomalous regions, even if the exact points of anomaly onset/offset are slightly misaligned.

## Data Loading
Use the provided `UCRAnomalyDataset` class in `anomaly_dataset.py` to load datasets:

```python
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class UCRAnomalyDataset:
    def __init__(self, name: str, data_path: str = "home/data"):
        """
        Load a single UCR anomaly detection dataset.

        Args:
            name: Name of the dataset (e.g., "186_UCR_Anomaly_resperation1")
            data_path: Path to the directory containing datasets
        """
        self.name = name
        self.data_path = Path(data_path)

        self.train_data = self._load_train_data()
        self.test_data = self._load_test_data()
        self.sample_submission = self._load_sample_submission()

    def _load_train_data(self) -> np.ndarray:
        """Load training data for this dataset
        Contains all the data points in the training set, no anomalies in it
        """
        train_path = self.data_path / f"{self.name}_train.csv"
        return pd.read_csv(train_path).to_numpy().flatten()

    def _load_test_data(self) -> np.ndarray:
        """Load test data for this dataset
        Contains all the data points in the test set there is an anomaly in it
        """
        test_path = self.data_path / f"{self.name}_test.csv"
        return pd.read_csv(test_path).to_numpy().flatten()

    def _load_sample_submission(self) -> np.ndarray:
        """Load sample submission for this dataset
        Contains the anomaly labels for the test set
        """
        sample_sub_path = self.data_path / "sample_submission.h5"

        with h5py.File(sample_sub_path, "r") as f:
            return f["submission"][self.name][:]
```

Here is an example of how to load a single dataset:
```python
# Load a single dataset
dataset = UCRAnomalyDataset(name="187_UCR_Anomaly_resperation2")

# Access the data
train_data = dataset.train_data  # Normal patterns (no anomalies)
test_data = dataset.test_data    # Contains an anomaly
sample_submission = dataset.sample_submission  # Contains the sample submission format (anomaly scores at each time step)
```

## Submission Format
For each test dataset, you must predict anomaly scores for each data point. Higher scores should indicate a higher likelihood of being an anomaly. Save your predictions in an h5 file with the same structure as the provided sample submission. Save your predictions in the `submission` group. The key should be the dataset name. The value should be a numpy array of shape (length of the test set,). File name should be `submission.h5`.

Good luck!
