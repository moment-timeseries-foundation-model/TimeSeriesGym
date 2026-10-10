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
        branch="moirai-evaluation",
        commit="e69271d9e7b85c8ac110868b1b30dd857cd14850",
    )

    # Remove all .json and .csv files from the cloned repository
    for file in (public / "context-is-key-forecasting").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()
