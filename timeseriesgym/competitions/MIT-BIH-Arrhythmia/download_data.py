import argparse
import os
from pathlib import Path

from tqdm import tqdm

from timeseriesgym.registry import registry
from timeseriesgym.utils import compress

# TODO(mononito): Maybe this should be moved elsewhere?


def download_mitdb(download_dir: Path, force: bool = False, **kwargs):
    """
    Download the MIT-BIH Arrhythmia Database from PhysioNet.

    This function downloads all record files from the MIT-BIH Arrhythmia Database,
    which contains 48 half-hour excerpts of two-channel ambulatory ECG recordings.
    Each record includes three file types:
        - .dat: Binary signal data
        - .hea: Header file with metadata
        - .atr: Annotation file with beat labels

    Returns:
        None

    Source:
        Adapted from https://github.com/physhik/ecg-mit-bih/blob/master/src/data.py

    Example:
        >>> download_mitdb()
    """
    # File extensions to download for each record
    extensions = ["dat", "hea", "atr"]

    # Record numbers available in the MIT-BIH Arrhythmia Database
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

    # Base URL for the PhysioNet MIT-BIH database
    base_url = "https://physionet.org/physiobank/database/mitdb/"
    competition_id = "MIT-BIH-Arrhythmia"
    data_path = download_dir / competition_id / "raw"

    # Create dataset directory if it doesn't exist
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    # Download each file with progress bar
    for record in tqdm(record_numbers, desc="Downloading Records"):
        for extension in extensions:
            file_url = f"{base_url}{record}.{extension}"

            if force or not data_path.joinpath(f"{record}.{extension}").exists():
                download_cmd = f"cd {data_path} && curl -O {file_url}"
                os.system(download_cmd)

    # Zip the downloaded files
    compress(data_path, data_path.parent / "MIT-BIH-Arrhythmia.zip")

    return data_path.parent / "MIT-BIH-Arrhythmia.zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse arguments to download MIT-BIH Arrhythmia Database."
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default=registry.get_data_dir(),
    )
    parser.add_argument(
        "--force",
        help="Force download of the dataset.",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    download_mitdb(download_dir=Path(args.data_dir), force=args.force)
