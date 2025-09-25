from pathlib import Path

import h5py
import numpy as np
import wfdb
from scipy.signal import find_peaks
from sklearn import preprocessing
from tqdm import tqdm

from timeseriesgym.utils import get_logger

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Preprocess the MIT-BIH ECG data for machine learning.

    This function processes ECG data from the MIT-BIH Arrhythmia Database,
    extracting features and labels for each heartbeat.

    The function:
    1. Reads ECG signals from database records
    2. Normalizes the signals
    3. Detects peaks (R-peaks) in the ECG signals
    4. Extracts fixed-size segments around each peak
    5. Labels each segment according to its annotation
    6. Adds synthetic noise samples to improve robustness
    7. Saves processed data as HDF5 files

    Source:
        Adapted from https://github.com/physhik/ecg-mit-bih/blob/master/src/data.py

    Returns:
        None: Data is saved to HDF5 files in the dataset directory
    """
    # Record numbers available in the MIT-BIH Arrhythmia Database
    seed = 42
    np.random.seed(seed)
    record_numbers = [
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "121",
        "122",
        "123",
        "124",
        "200",
        "201",
        "202",
        "203",
        "205",
        "207",
        "208",
        "209",
        "210",
        "212",
        "213",
        "214",
        "215",
        "217",
        "219",
        "220",
        "221",
        "222",
        "223",
        "228",
        "230",
        "231",
        "232",
        "233",
        "234",
    ]
    segment_len = 256  # Length of each ECG segment

    # Define training and testing sets
    test_records = ["101", "105", "114", "118", "124", "201", "210", "217"]
    train_records = [record for record in record_numbers if record not in test_records]
    sample_to = {
        "100": 206165,
        "102": 588458,
    }  # Records 100 and 102 have different lengths for different channels

    def process_dataset(record_set) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Process a set of records and save the extracted features and labels.

        Args:
            record_set (list): List of record numbers to process
        """
        # Heart beat classes to detect
        classes = ["N", "V", "/", "A", "F", "~"]  # Normal, PVC, Paced, Atrial, Fusion, Noise
        n_classes = len(classes)

        # Initialize data structures
        data_dict = {"lead_1": [], "lead_2": []}
        labels = []

        def process_records():
            """Process each record to extract features and labels."""

            for record_num in tqdm(record_set, desc="Processing Records"):
                # Read the record
                record_path = raw / record_num
                record = wfdb.rdrecord(
                    str(record_path), smooth_frames=True, sampto=sample_to.get(record_num, None)
                )

                # Normalize signals
                signal_0 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 0])).tolist()
                signal_1 = preprocessing.scale(np.nan_to_num(record.p_signal[:, 1])).tolist()

                # Find peaks for segmentation
                peaks, _ = find_peaks(signal_0, distance=150)

                # Process each detected peak
                for peak in tqdm(
                    peaks[1:-1], desc=f"Processing peaks in record {record_num}", leave=False
                ):
                    # Define segment boundaries around the peak
                    start, end = peak - segment_len // 2, peak + segment_len // 2

                    # Get annotations for this segment
                    ann = wfdb.rdann(
                        str(record_path),
                        extension="atr",
                        sampfrom=start,
                        sampto=end,
                        return_label_elements=["symbol"],
                    )

                    # Process annotation if it's a single valid class
                    ann_symbol = ann.symbol
                    if (
                        len(ann_symbol) == 1
                        and ann_symbol[0] in classes
                        and (ann_symbol[0] != "N" or np.random.random() < 0.15)
                    ):
                        # Create one-hot encoding for the class
                        label = [0] * n_classes
                        label[classes.index(ann_symbol[0])] = 1

                        # Store the segment and its label
                        labels.append(label)
                        data_dict["lead_1"].append(signal_0[start:end])
                        data_dict["lead_2"].append(signal_1[start:end])

        # Process all records
        process_records()

        # TODO: Add synthetic noise samples

        # Convert data to numpy arrays
        for channel in data_dict.keys():
            data_dict[channel] = np.array(data_dict[channel])
        labels = np.array(labels)

        return data_dict, labels

    # Process data
    train_data_dict, train_labels = process_dataset(train_records)

    # Save the train dataset and labels
    with h5py.File(public / "train_features.h5", "w") as dataset_file:
        group = dataset_file.create_group("train")
        for channel in train_data_dict.keys():
            group.create_dataset(channel, data=train_data_dict[channel])

    with h5py.File(public / "train_labels.h5", "w") as labels_file:
        labels_file.create_dataset("labels", data=train_labels)

    logger.info("Saved train features and labels!")

    # Process test data
    test_data_dict, test_labels = process_dataset(test_records)

    with h5py.File(public / "test_features.h5", "w") as dataset_file:
        group = dataset_file.create_group("test")
        for channel in test_data_dict.keys():
            group.create_dataset(channel, data=test_data_dict[channel])

    with h5py.File(private / "test_labels.h5", "w") as labels_file:
        labels_file.create_dataset("labels", data=test_labels)

    logger.info("Saved test features and labels!")

    sample_test_labels = np.zeros_like(test_labels)
    sample_test_labels[:, 0] = 1  # Predict all normal heartbeats

    with h5py.File(public / "sample_submission.h5", "w") as labels_file:
        labels_file.create_dataset("submission", data=sample_test_labels)

    logger.info("Saved sample submission!")
