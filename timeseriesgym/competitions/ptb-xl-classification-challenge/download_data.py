import argparse
import ast
import json
import os
import subprocess
from itertools import compress
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tqdm import tqdm

from timeseriesgym.utils import compress as compress_zip
from timeseriesgym.utils import extract


def is_directory(path):
    extensions = [".pth", ".txt", ".json", ".yaml"]

    for ext in extensions:
        if ext in path:
            return False
    return True


def make_dir_if_not_exists(path, verbose=True):
    if not is_directory(path):
        path = path.split(".")[0]
    if not os.path.exists(path=path):
        os.makedirs(path)
        if verbose:
            print(f"Making directory: {path}...")
    return True


def load_paths(basepath, fs):
    """File Names to Load"""
    folders1 = os.listdir(basepath)
    pth_to_folders = [
        folder for folder in folders1 if "records%i" % fs in folder
    ]  # records100 contains 100Hz data
    pth_to_folders = [
        os.path.join(pth_to_folder, fldr)
        for pth_to_folder in pth_to_folders
        for fldr in os.listdir(os.path.join(basepath, pth_to_folder))
        if not fldr.endswith("index.html")
    ]
    files = [
        os.path.join(pth_to_folder, fldr)
        for pth_to_folder in pth_to_folders
        for fldr in os.listdir(os.path.join(basepath, pth_to_folder))
    ]
    paths_to_files = [
        os.path.join(basepath, file.split(".hea")[0]) for file in files if ".hea" in file
    ]
    return paths_to_files


def modify_df(basepath, output_type="single"):
    """Creates a dataframe with patient information, reports, data file paths and labels"""
    """ Database with Patient-Specific Info """
    df = pd.read_csv(os.path.join(basepath, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    """ Database with Label Information """
    codes_df = pd.read_csv(os.path.join(basepath, "scp_statements.csv"), index_col=0)

    if output_type == "single":
        encoder = LabelEncoder()
    elif output_type == "multi":
        encoder = MultiLabelBinarizer()

    def aggregate_diagnostic(y_dic):
        """Map Label To Diffeent Categories"""
        tmp = []
        for key in y_dic.keys():
            if key in diag_agg_df.index:
                c = diag_agg_df.loc[key].diagnostic_class
                if str(c) != "nan":
                    tmp.append(c)
        return list(set(tmp))

    aggregation_df = codes_df
    diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]

    """ Obtain Superdiagnostic Label(s) """
    df["superdiagnostic"] = df.scp_codes.apply(aggregate_diagnostic)
    """ Obtain Number of Superdiagnostic Label(s) Per Recording """
    df["superdiagnostic_len"] = df.superdiagnostic.apply(lambda x: len(x))

    """ Return Histogram of Each Label """
    min_samples = 0
    counts = pd.Series(np.concatenate(df.superdiagnostic.values)).value_counts()
    counts = counts[counts > min_samples]

    """ Obtain Encoded Label as New Column """
    if output_type == "single":
        df = df[df.superdiagnostic_len == 1]  # ==1 OR > 0
        df.superdiagnostic = df.superdiagnostic.apply(lambda entry: entry[0])
        encoder.fit(df.superdiagnostic.values)
        df["superdiagnostic_label"] = encoder.transform(df.superdiagnostic.values)
    elif output_type == "multi":
        df = df[df.superdiagnostic_len > 0]
        # encoder.fit(list(map(lambda entry: [entry], counts.index.values)))
        encoder.fit([[entry] for entry in counts.index.values])
        multi_hot_encoding_df = pd.DataFrame(
            encoder.transform(df.superdiagnostic.values),
            index=df.index,
            columns=encoder.classes_.tolist(),
        )
        df = pd.merge(df, multi_hot_encoding_df, on=df.index).drop(["key_0"], axis=1)

    return df


def obtain_phase_to_paths_dict(df, paths_to_files, phase_to_pids=None):
    """Creates a mapping between the phase (train, val, test)
    and data files belonging to that phase
    """
    train_fold = np.arange(0, 8)
    val_fold = [9]
    test_fold = [10]
    phases = ["train", "val", "test"]
    folds = [train_fold, val_fold, test_fold]

    """ Obtain Patient IDs """
    if phase_to_pids is None:
        phase_to_pids = {}
        for phase, fold in zip(phases, folds, strict=False):
            current_ecgid = df[
                df.strat_fold.isin(fold)
            ].index.tolist()  # index is ecg_id by default when loading csv
            current_ecgid = [int(entry) for entry in current_ecgid]
            phase_to_pids[phase] = current_ecgid
    else:
        phase_to_pids = phase_to_pids

    # paths_to_ids = list(map(lambda path: int(path.split("/")[-1].split("_")[0]), paths_to_files))
    paths_to_ids = [int(path.split("/")[-1].split("_")[0]) for path in paths_to_files]
    paths_to_ids_df = pd.Series(paths_to_ids)
    """ Obtain Paths For Each Phase """
    phase_to_paths = {}

    for phase, pids in phase_to_pids.items():
        """ Obtain Paths For All Leads """
        paths = list(compress(paths_to_files, paths_to_ids_df.isin(pids).tolist()))
        # """ Assign Paths and Leads Labels """
        phase_to_paths[phase] = paths
    # pdb.set_trace()
    return phase_to_paths


def load_ptbxl_data(phase_to_paths, df, phase, output_type="single"):
    """Load PTB-XL Data"""
    paths = phase_to_paths[phase]  # this is the most essential part

    """ Obtain IDs """
    # ecg_ids = list(map(lambda path: int(path.split("/")[-1].split("_")[0]), paths))
    ecg_ids = [int(path.split("/")[-1].split("_")[0]) for path in paths]
    print("[INFO] Obtaining Labels")

    """ Obtain Labels """
    if output_type == "single":
        labels = np.asarray(
            [df[df.index == id_entry].superdiagnostic_label.iloc[0] for id_entry in tqdm(ecg_ids)]
        )
    elif output_type == "multi":
        labels = np.asarray(
            [df[df.index == id_entry].iloc[0][-5:].tolist() for id_entry in tqdm(ecg_ids)]
        )

    print("[INFO] Obtaining Text Reports")
    """ Obtain Report Text For Each Entry """
    text_reports = np.asarray(
        [df[df.index == id_entry].report.iloc[0] for id_entry in tqdm(ecg_ids)]
    )

    return text_reports, labels, ecg_ids


class PTBXLDataset:
    def __init__(self, config, phase):
        self.phase = phase
        self.bands = list(range(12))
        self.fs = config.fs
        self.output_type = config.output_type

        print("[INFO] Loading PTB-XL")

        # Load path to the data and get dataframe with patient attributes,
        # data path, labels and expert interpretations
        paths_to_files = load_paths(config.base_path, fs=self.fs)
        self.df = modify_df(config.base_path, output_type=self.output_type)

        print("[INFO] Modified database")
        # Create a mapping between data files and the train, test, val split they belong to
        if getattr(config, "path_to_pids", None) and os.path.exists(config.path_to_pids):
            phase_to_pids = json.load(open(config.path_to_pids))
            self.phase_to_paths = obtain_phase_to_paths_dict(
                self.df, paths_to_files, phase_to_pids=phase_to_pids
            )
        else:
            self.phase_to_paths = obtain_phase_to_paths_dict(self.df, paths_to_files)

        print("[INFO] Extracted paths")
        self._texts, self._labels, self._ecg_ids = load_ptbxl_data(
            self.phase_to_paths, self.df, self.phase, self.output_type
        )

        print("[INFO] Loading ECG Signals")
        ecg_input_paths = self.phase_to_paths[self.phase]
        self._timeseries = []

        self._timeseries = Parallel(n_jobs=1)(
            delayed(self.get_timeseries)(ecg_input_path) for ecg_input_path in tqdm(ecg_input_paths)
        )
        self._length = len(self._timeseries)
        self.n_classes = len(np.unique(self._labels))

    def get_timeseries(self, ecg_input_path):
        ecg_signal, _ = wfdb.rdsamp(ecg_input_path, channels=self.bands)
        return ecg_signal.T


def main(args):
    competition_dir = Path(args.data_dir) / "ptb-xl-classification-challenge"
    os.makedirs(competition_dir, exist_ok=True)
    tmp_dir = competition_dir / "tmp"  # temporary directory to save the csv files
    os.makedirs(tmp_dir, exist_ok=True)

    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

    wget_command = [
        "wget",
        "-P",
        competition_dir,  # Specify download directory
        url,
    ]

    subprocess.run(wget_command, check=True)
    print(f"[INFO] downloaded data to {competition_dir}")

    # extract the downloaded zip file
    destination = competition_dir / "ptb-xl-base"
    os.makedirs(destination, exist_ok=True)
    extract(
        competition_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
        destination,
    )
    print(f"[INFO] extracted data to {destination}")

    # remove the zip file
    os.system(
        f"rm {competition_dir}/"
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    )
    print(
        f"[INFO] removed {competition_dir}/"
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    )

    # set base path and load the dataset
    args.base_path = (
        destination / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )
    train_dataset = PTBXLDataset(args, "train")
    val_dataset = PTBXLDataset(args, "val")
    test_dataset = PTBXLDataset(args, "test")

    # convert dataset to csv format
    test_label_df = pd.DataFrame({"label": test_dataset._labels})
    sample_submissions = pd.DataFrame({"label": [0] * len(test_dataset._labels)})

    with h5py.File(os.path.join(tmp_dir, "train_df.h5"), "w") as f:
        f.create_dataset("signals", data=train_dataset._timeseries)
        f.create_dataset("labels", data=train_dataset._labels)
    with h5py.File(os.path.join(tmp_dir, "val_df.h5"), "w") as f:
        f.create_dataset("signals", data=val_dataset._timeseries)
        f.create_dataset("labels", data=val_dataset._labels)
    with h5py.File(os.path.join(tmp_dir, "test_df.h5"), "w") as f:
        f.create_dataset("signals", data=test_dataset._timeseries)
    test_label_df.to_csv(os.path.join(tmp_dir, "test_label_df.csv"))
    sample_submissions.to_csv(os.path.join(tmp_dir, "sample_submission.csv"))

    # make sure there are no additional files in the tmp directory except from the csv files
    for file in os.listdir(tmp_dir):
        if file not in [
            "train_df.h5",
            "val_df.h5",
            "test_df.h5",
            "test_label_df.csv",
            "sample_submission.csv",
        ]:
            raise ValueError(f"[ERROR] unexpected file {file} in {tmp_dir}")

    # compress the files
    zipfile_path = competition_dir / "ptb-xl-classification-challenge_test.zip"
    compress_zip(tmp_dir, zipfile_path)
    print(f"[INFO] compressed files to {zipfile_path}")

    # remove tmp directory
    os.system(f"rm -r {tmp_dir}")
    print(f"[INFO] removed {tmp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ptbxl dataset config -- don't have to change this
    parser.add_argument("--fs", type=int, default=100, help="Sampling frequency")
    parser.add_argument("--output_type", type=str, default="single")
    parser.add_argument("--code_of_interest", type=str, default="diagnostic_class")

    # other config
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to save the preprocessed zip file"
    )
    args = parser.parse_args()

    main(args)
