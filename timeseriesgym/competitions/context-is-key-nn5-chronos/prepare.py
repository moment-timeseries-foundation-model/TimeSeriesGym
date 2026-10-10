from pathlib import Path

from timeseriesgym.utils import clone_repository


def prepare(raw: Path, public: Path, private: Path):
    sample_submission = raw / "sample_submission.pkl"
    solution = raw / "solution.pkl"

    solution.rename(private / "solution.pkl")
    sample_submission.rename(public / "sample_submission.pkl")

    # Download the cik GitHub repository
    clone_repository(
        "https://github.com/moment-timeseries-foundation-model/context-is-key-forecasting.git",
        public / "context-is-key-forecasting",
        branch="nn5-chronos",
        commit="5a0fdc0b1f9facad9324b4aadda6df73c7851115",
    )

    # Remove all .json and .csv files from the cloned repository
    for file in (public / "context-is-key-forecasting").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()
