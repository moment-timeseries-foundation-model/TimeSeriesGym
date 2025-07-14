import logging
import os
import sys
import zipfile
from pathlib import Path

import gdown
import h5py
import numpy as np


def prepare(raw: Path, public: Path, private: Path):
    # Download the long-term forecast dataset from Google Drive by TSL authors
    url = "https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322"
    gdown.download_folder(url=url, output=str(raw), quiet=False, use_cookies=False)

    # Unzip the downloaded files in the long_term_forecast directory
    zip_dir = raw
    for zip_path in zip_dir.glob("*.zip"):
        extract_dir = zip_dir  # Create folder with same name as zip (without .zip)

        logging.info(f"Unzipping {zip_path.name} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # move the folder to public
        extracted_folder = extract_dir / zip_path.stem
        extracted_folder.rename(public / zip_path.stem)

    # Download the tsl GitHub repository
    os.system(
        f"cd {public}; git clone --branch long-horizon-forecast --single-branch https://github.com/raycai420/Time-Series-Library.git"
    )

    # Remove all .json and .csv files from the cloned repository
    for file in (public / "Time-Series-Library").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()

    sys.path.insert(0, str(public / "Time-Series-Library"))
    from data_provider.data_factory import data_provider

    # 4. Run the data loading logic
    def get_true_array(ds_name, pred_len=96, seq_len=96, root_path=None, data_path=None):
        class Arg:
            def __init__(self):
                self.data = "custom" if "ett" not in ds_name.lower() else ds_name
                self.pred_len = pred_len
                self.seq_len = seq_len
                self.root_path = root_path
                self.features = "M"
                self.embed = "-"
                self.batch_size = 512
                self.freq = "-"
                self.task_name = "long_term_forecast"
                self.data_path = data_path
                self.label_len = 0
                self.target = "OT"
                self.seasonal_patterns = None
                self.num_workers = 1

        args = Arg()
        data_set, data_loader = data_provider(args, "test")

        trues = []
        for _, (_, batch_y, _, _) in enumerate(data_loader):
            batch_y = batch_y[:, -args.pred_len :, :]
            trues.append(batch_y)

        trues = np.concatenate(trues, axis=0)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        return trues

    ds = {}
    for root_path in public.iterdir():
        # loop through all .csv files in root_path
        for file in root_path.glob("*.csv"):
            ds_name = file.stem
            data_path = file.name
            trues = get_true_array(
                ds_name, pred_len=96, seq_len=96, root_path=root_path, data_path=data_path
            )

            trues = trues.astype(np.float16)
            ds[ds_name] = trues
            logging.info(f"Processed {ds_name} with shape {trues.shape}")

    # Directory containing your .npy files
    output_h5_file = private / "test.h5"
    with h5py.File(output_h5_file, "w") as h5f:
        submission_group = h5f.create_group("labels")  # Top-level dummy layer
        for ds_name, data in ds.items():
            # Create a dataset for each .npy file
            submission_group.create_dataset(ds_name, data=data)

    sample_submission_h5 = public / "sample_submission.h5"
    with h5py.File(sample_submission_h5, "w") as h5f:
        submission_group = h5f.create_group("submission")  # Top-level dummy layer
        for ds_name, data in ds.items():
            # Create a dataset for each .npy file
            dummy = np.zeros(data.shape)
            submission_group.create_dataset(ds_name, data=dummy)
