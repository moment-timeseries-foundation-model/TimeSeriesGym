import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timeseriesgym.utils import import_fn, read_csv, read_parquet


class MissingnessSimulator:
    def __init__(
        self,
        name: str,
        missingness_fn: str,
        file_names: list | None = None,
        dir_names: list | None = None,
        ignore_columns: list | None = None,
        missingness_ratio: float = 0.0,
    ) -> None:
        """
        Initializes the MissingnessSimulator class.

        Args:
            name: The name of the missingness simulator.
            missingness_fn: The import path of the function to simulate missingness.
            file_names: A list of file names to apply missingness simulation.
            dir_names: A list of directory names containing files for missingness simulation.
            ignore_columns: A list of column names to exclude from missingness simulation.
            missingness_ratio: The ratio of missing values to introduce (between 0 and 1).
        """
        self.name = name
        self.missingness_fn = import_fn(missingness_fn)
        self.file_names = [] if file_names is None else file_names
        self.dir_names = [] if dir_names is None else dir_names
        self.ignore_columns = [] if ignore_columns is None else ignore_columns
        self.missingness_ratio = missingness_ratio

        assert isinstance(self.name, str), "MissingnessSimulator name must be a string."
        assert is_valid_missingness_fn(
            self.missingness_fn
        ), "Invalid arguments of `missingness_fn`."
        assert isinstance(self.file_names, list), "file_names must be a list."
        assert isinstance(self.dir_names, list), "dir_names must be a list."
        assert isinstance(self.ignore_columns, list), "ignore_columns must be a list."
        assert isinstance(
            self.missingness_ratio, float
        ), "missingness_ratio must be a float number."
        assert len(self.name) > 0, "MissingnessSimulator name cannot be empty."
        assert (
            len(self.file_names) > 0 or len(self.dir_names) > 0
        ), "At least one of file_names or dir_names must be specified."
        assert 0 < self.missingness_ratio < 1, "missingness_ratio must be between 0 and 1."

    @staticmethod
    def from_dict(data: dict) -> "MissingnessSimulator":
        return MissingnessSimulator(**data)

    def __call__(self, public: Path) -> None:
        # find all files for which missingness will be simulated
        all_file_paths = []
        if self.file_names:
            file_paths = [public / f for f in self.file_names]
            all_file_paths.extend(file_paths)
        if self.dir_names:
            for dir_name in self.dir_names:
                file_paths = [f for f in (public / dir_name).rglob("*") if f.is_file()]
                all_file_paths.extend(file_paths)

        for file_path in all_file_paths:
            assert file_path.is_file(), f"File {file_path} does not exist."

            train_df = self.missingness_fn(
                file_path=file_path,
                ignore_columns=self.ignore_columns,
                missingness_ratio=self.missingness_ratio,
            )

            if file_path.suffix.lower() == ".csv":
                train_df.to_csv(file_path, index=False)
            elif file_path.suffix.lower() == ".parquet":
                train_df.to_parquet(file_path, index=False)


def is_valid_missingness_fn(missingness_fn: Any) -> bool:
    """Checks if the `missingness_fn` takes three arguments: `file_path`, `ignore_columns` and
    `missingness_ratio`, in that order."""

    try:
        sig = inspect.signature(missingness_fn)
    except (TypeError, ValueError):
        return False

    actual_params = list(sig.parameters.keys())
    expected_params = ["file_path", "ignore_columns", "missingness_ratio"]

    return actual_params == expected_params


def simulate_random_missingness(
    file_path: Path,
    ignore_columns: list,
    missingness_ratio: float,
) -> "pd.DataFrame":
    ext = file_path.suffix.lower()
    assert ext in [".csv", ".parquet"], "The file must be a csv or a parquet file."

    if ext == ".csv":
        train_df = read_csv(file_path)
    elif ext == ".parquet":
        train_df = read_parquet(file_path)

    for i, col_name in enumerate(train_df.columns):
        if col_name not in ignore_columns:
            num_missing = int(len(train_df) * missingness_ratio)

            np.random.seed(i + 42)
            missing_indices = np.random.choice(train_df.index, num_missing, replace=False)
            train_df.loc[missing_indices, col_name] = np.nan

    return train_df
