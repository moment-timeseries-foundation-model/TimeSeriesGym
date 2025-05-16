import os
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path):
    truth = raw / "test.h5"
    truth.rename(private / "test.h5")

    # create sample submission as save structure of the h5
    sample_submission = raw / "sample_submission.h5"
    sample_submission.rename(public / "sample_submission.h5")

    # move all the folders to public
    for folder in raw.iterdir():
        if folder.is_dir():
            folder.rename(public / folder.name)

    # Download the tsl GitHub repository
    os.system(
        f"cd {public}; git clone --branch long-horizon-forecast --single-branch https://github.com/raycai420/Time-Series-Library.git"
    )

    # Remove all .json and .csv files from the cloned repository
    for file in (public / "time-series-library").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()
