import logging
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_ucr_anomaly_detection_dataset(datasets_name: str):
    underscore_split = datasets_name.split("_")
    dataset_dict = {
        "name": underscore_split[0]
        + "_"
        + underscore_split[1]
        + "_"
        + underscore_split[2]
        + "_"
        + underscore_split[3],
        "end_training_data": int(underscore_split[4]),
        "anomaly_begin": int(underscore_split[5]),
        "anomaly_end": int(underscore_split[6]),
    }
    return (int(underscore_split[0]), dataset_dict)


def prepare(raw: Path, public: Path, private: Path) -> None:
    random_seed = 0
    datasets_path = (
        raw
        / "AnomalyDatasets_2021"
        / "UCR_TimeSeriesAnomalyDatasets2021"
        / "FilesAreInHere"
        / "UCR_Anomaly_FullData"
    )
    np.random.seed(random_seed)
    # instead of using commans they use tabs and need manual handling
    messed_up_datasets = [204, 205, 206, 207, 208, 225, 226, 242, 243]

    dataset_names = []

    sample_submission_h5_file = public / "sample_submission.h5"
    labels_h5_file = private / "labels.h5"
    datasets_for_processing_original = Path(
        "timeseriesgym/competitions/ucr-anomaly-detection/datasets.txt"
    )
    anomaly_dataset_code_original = Path(
        "timeseriesgym/competitions/ucr-anomaly-detection/anomaly_dataset.py"
    )

    shutil.copy2(datasets_for_processing_original, public / "datasets.txt")
    shutil.copy2(anomaly_dataset_code_original, public / "anomaly_dataset.py")

    with open(public / "datasets.txt") as f:
        dataset_names_required = [line.strip() for line in f.readlines()]

    os.makedirs(public / "train", exist_ok=True)
    os.makedirs(public / "test", exist_ok=True)

    with (
        h5py.File(sample_submission_h5_file, "w") as f_sub,
        h5py.File(labels_h5_file, "w") as f_label,
    ):
        submission_group = f_sub.create_group("submission")
        labels_group = f_label.create_group("labels")

        for file in sorted(os.listdir(datasets_path)):
            dataset_number, dataset_dict = parse_ucr_anomaly_detection_dataset(file.split(".")[0])
            dataset_name = dataset_dict["name"]
            if dataset_name not in dataset_names_required:
                # skip if not in the required list
                continue
            dataset_names.append(dataset_name)

            if dataset_number in messed_up_datasets:
                with open(datasets_path / file) as f:
                    data = f.read().split()
                    data = [float(x) for x in data]
                    df = pd.DataFrame(data)
            else:
                df = pd.read_csv(datasets_path / file, header=None)

            training_df = df.iloc[: dataset_dict["end_training_data"]]
            test_df = df.iloc[dataset_dict["end_training_data"] :]
            anomaly_begin = int(dataset_dict["anomaly_begin"]) - int(
                dataset_dict["end_training_data"]
            )
            anomaly_end = int(dataset_dict["anomaly_end"]) - int(dataset_dict["end_training_data"])

            test_labels = np.zeros(len(test_df))
            sample_submission = np.zeros(len(test_df))
            test_labels[anomaly_begin:anomaly_end] = 1

            train_dataset_file = public / "train" / f"{dataset_name}.csv"
            test_file = public / "test" / f"{dataset_name}.csv"

            training_df.to_csv(train_dataset_file, index=False)
            test_df.to_csv(test_file, index=False)

            submission_group.create_dataset(dataset_name, data=sample_submission)
            labels_group.create_dataset(dataset_name, data=test_labels)

    logger.info(f"Prepared {len(dataset_names)} datasets")
