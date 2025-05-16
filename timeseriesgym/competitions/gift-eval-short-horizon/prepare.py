import logging
import shutil
from pathlib import Path

import h5py
import numpy as np


def prepare(raw: Path, public: Path, private: Path):
    datasets_path = Path("timeseriesgym/competitions/gift-eval-short-horizon/datasets.txt")
    dataset_prepare_path = Path("timeseriesgym/competitions/gift-eval-short-horizon/dataset.py")

    # copy over to public
    shutil.copy2(datasets_path, public / datasets_path.name)
    logging.info(f"Copying {datasets_path} to {public / datasets_path.name}")

    # copy over to public
    shutil.copy2(dataset_prepare_path, public / dataset_prepare_path.name)
    logging.info(f"Copying {dataset_prepare_path} to {public / dataset_prepare_path.name}")

    with open(public / datasets_path.name) as f:
        dataset_names = [line.strip() for line in f.readlines()]

    logging.info(f"Dataset names: {dataset_names}")

    # make new test.h5 from the raw data
    with (
        h5py.File(raw / "test.h5", "r") as original_file,
        h5py.File(private / "test.h5", "w") as subset_file,
    ):
        # Load the original data
        original_data = original_file["label"][()]

        # Get all dataset names from the original file
        all_dataset_names = original_data.dtype.names

        # Filter for only the datasets in our subset
        valid_subset_names = [name for name in dataset_names if name in all_dataset_names]

        # Create a new compound dtype with only the subset fields
        subset_dt = []
        subset_data = {}

        # Collect data and prepare dtype
        for name in valid_subset_names:
            # Get shape from original data
            shape = original_data[name][0].shape
            subset_dt.append((name, np.float32, shape))
            subset_data[name] = original_data[name][0]

        # Create the compound dtype
        subset_dt = np.dtype(subset_dt)

        # Create and fill subset data array
        data = np.zeros(1, dtype=subset_dt)
        for name in valid_subset_names:
            data[name][0] = subset_data[name]

        # Save to the new file
        subset_file.create_dataset("labels", data=data)
        logging.info(f"Created {private / 'test.h5'} with subset data.")

    # create a sample submission file of the same shape as the test.h5 file
    with h5py.File(public / "sample_submission.h5", "w") as sample_file:
        # Create a dataset with the same shape as the test.h5 file
        sample_data = np.zeros(1, dtype=subset_dt)
        sample_file.create_dataset("submission", data=sample_data)
        logging.info(f"Created {public / 'sample_submission.h5'} with sample submission data.")

    # move dataset to public
    for ds_name in dataset_names:
        ds_name = ds_name.split("/")[0]
        item = raw / ds_name
        item.rename(public / ds_name)

    # for item in raw.iterdir():
    #     # Skip the test.h5 file
    #     if item.name == "test.h5":
    #         continue

    #     if item.is_file():
    #         item.rename(public / item.name)
    #     elif item.is_dir():
    #         item.rename(public / item.name)
