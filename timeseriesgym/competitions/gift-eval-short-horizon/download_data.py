import argparse
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from dataset import Dataset, Term

from timeseriesgym.utils import compress


def main(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    gift_eval_path = Path(data_dir) / "gift_eval_raw"
    os.makedirs(gift_eval_path, exist_ok=True)

    # Construct the command
    command = (
        f"huggingface-cli download Salesforce/GiftEval "
        f"--repo-type=dataset --local-dir {gift_eval_path}"
    )

    # Execute the command
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("[INFO] Download GIFT-Eval successful to path:", data_dir)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error downloading the dataset: {e}")
        print(f"[ERROR] Error output: {e.stderr}")

    # Get all subdirectories (dataset names) in the GIFT_EVAL path
    dataset_names = []
    for dataset_dir in gift_eval_path.iterdir():
        if dataset_dir.name.startswith("."):
            continue
        if dataset_dir.is_dir():
            freq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            if freq_dirs:
                for freq_dir in freq_dirs:
                    dataset_names.append(f"{dataset_dir.name}/{freq_dir.name}")
            else:
                dataset_names.append(dataset_dir.name)

    print("[INFO] Available datasets in GIFT_EVAL:")
    for name in sorted(dataset_names):
        print(f"- {name}")

    # create dt
    dt = []
    test_data = {}

    for dataset_name in dataset_names:
        dataset = Dataset(
            dataset_name,
            term=Term.SHORT,
            to_univariate=False,
            storage_path=gift_eval_path,
        )
        test_array = []
        for sample in dataset.test_data.label:
            test_array.append(sample["target"])
        test_array = np.array(test_array)
        dt.append((dataset_name, np.float32, test_array.shape))
        test_data[dataset_name] = test_array

    dt = np.dtype(dt)

    # save h5
    with h5py.File(gift_eval_path / "test.h5", "w") as f:
        data = np.zeros(1, dtype=dt)
        for dataset_name in dataset_names:
            data[dataset_name][0] = np.array(test_data[dataset_name])

        f.create_dataset("label", data=data)

    print("[INFO] Test data saved to h5 file.")

    # Compress the data and remove the original file
    competition_dir = Path(data_dir) / "gift-eval-short-horizon"
    os.makedirs(competition_dir, exist_ok=True)
    compress(gift_eval_path, competition_dir / "gift-eval-short-horizon.zip")
    print("[INFO] Data compressed and moved to competition directory.")

    # Remove the original data directory
    os.system(f"rm -rf {gift_eval_path}")
    print("[INFO] Original data directory removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the data for the GIFT evaluation competition."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to save the data")
    args = parser.parse_args()

    main(args.data_dir)
