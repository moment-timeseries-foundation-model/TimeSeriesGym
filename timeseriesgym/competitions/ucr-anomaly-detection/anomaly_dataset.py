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
