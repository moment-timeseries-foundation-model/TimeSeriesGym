# UCR Anomaly Detection

## Overview
This competition focuses on univariate time series anomaly detection using the UCR Time Series Anomaly Datasets. The collection includes 250 distinct datasets spanning various domains, each presenting a unique anomaly detection challenge; this competition uses the five datasets listed in `datasets.txt`.

**IMPORTANT:**
You have to solve datasets one by one, as they are provided in `datasets.txt` file. Each line is a dataset name. You can pass this name to the `UCRAnomalyDataset` class to load the dataset.

## Challenge Description
Anomaly detection in time series is a critical task in many fields including manufacturing, healthcare, finance, and infrastructure maintenance. Your goal is to identify anomalous segments in time series data by analyzing patterns and detecting deviations from normal behavior.

## Dataset
Each dataset consists of:
- A training set containing only normal patterns
- A test set containing one anomalous segment

The datasets vary in length, sampling frequency, and the nature of the anomalies they contain. Datasets that you need to process are specified in datasets.txt, each line is a dataset name. You can pass this name to the `UCRAnomalyDataset` class to load the dataset.

## Evaluation
Submissions are evaluated by the average precision (AP) of your anomaly scores, the area under the precision-recall curve, computed on each dataset's point labels and averaged over the datasets. No threshold is needed: only the order of your scores within a dataset matters.

### Average Precision
For each dataset, every point of the test set is labelled 1 if it lies in the anomalous segment and 0 otherwise. The evaluation works as follows:

1. For each dataset, your model provides an anomaly score for each data point (higher values indicate higher likelihood of anomaly)
2. Each distinct score is a threshold: the points with a score at or above it are predicted as anomalies. Precision is the share of these points that are labelled 1, and recall is the share of the labelled points that are among them
3. The dataset's score is the average precision: the precision at each threshold, weighted by the increase in recall from the previous threshold, summed over the thresholds (as `sklearn.metrics.average_precision_score(labels, scores)` computes it)
4. The overall score is the mean of the dataset scores over all datasets in `datasets.txt`

A dataset scores 1.0 if every point of its anomalous segment has a higher score than every other point. Points with equal scores are predicted together, so constant or random scores earn about the share of anomalous points, which is small. Every normal point that is ranked above an anomalous one lowers the precision, so give the highest scores to the points of the anomaly itself rather than to a wide window around it.

The scores must be finite numbers.

## Data Loading
Use the provided `UCRAnomalyDataset` class in `anomaly_dataset.py` to load datasets:

```python
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class UCRAnomalyDataset:
    def __init__(self, name: str, data_path: str = "/home/data"):
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
        train_path = self.data_path / "train" / f"{self.name}.csv"
        return pd.read_csv(train_path).to_numpy().flatten()

    def _load_test_data(self) -> np.ndarray:
        """Load test data for this dataset
        Contains all the data points in the test set there is an anomaly in it
        """
        test_path = self.data_path / "test" / f"{self.name}.csv"
        return pd.read_csv(test_path).to_numpy().flatten()

    def _load_sample_submission(self) -> np.ndarray:
        """Load sample submission for this dataset
        Contains zero-filled placeholder anomaly scores for the test set (the submission format)
        """
        sample_sub_path = self.data_path / "sample_submission.h5"

        with h5py.File(sample_sub_path, "r") as f:
            return f["submission"][self.name][:]
```

Here is an example of how to load a single dataset:
```python
# Load a single dataset
dataset = UCRAnomalyDataset(name="001_UCR_Anomaly_DISTORTED1sddb40")

# Access the data
train_data = dataset.train_data  # Normal patterns (no anomalies)
test_data = dataset.test_data    # Contains an anomaly
sample_submission = dataset.sample_submission  # Contains the sample submission format (anomaly scores at each time step)
```

## Submission Format
For each test dataset, you must predict anomaly scores for each data point. Higher scores should indicate a higher likelihood of being an anomaly. Save your predictions in an h5 file with the same structure as the provided sample submission. Save your predictions in the `submission` group. The key should be the dataset name. The value should be a numpy array of shape (length of the test set,). File name should be `submission.h5`.

Good luck!
