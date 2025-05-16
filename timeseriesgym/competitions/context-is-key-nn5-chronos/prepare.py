import os
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path):
    sample_submission = raw / "sample_submission.pkl"
    solution = raw / "solution.pkl"

    solution.rename(private / "solution.pkl")
    sample_submission.rename(public / "sample_submission.pkl")

    # Download the cik GitHub repository
    os.system(
        f"cd {public}; git clone --branch nn5-chronos --single-branch https://github.com/moment-timeseries-foundation-model/context-is-key-forecasting.git"
    )

    # Remove all .json and .csv files from the cloned repository
    for file in (public / "context-is-key-forecasting").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()
