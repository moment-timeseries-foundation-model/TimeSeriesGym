import shutil
from pathlib import Path

import numpy as np

from timeseriesgym.utils import clone_repository


def prepare(raw: Path, public: Path, private: Path):
    solution = raw / "pred.npy"
    solution = np.load(solution)
    sample_submission = np.zeros(solution.shape)

    np.save(private / "test.npy", solution)
    np.save(public / "sample_submission.npy", sample_submission)

    # Download the tsl GitHub repository
    clone_repository(
        "https://github.com/raycai420/Time-Series-Library.git",
        public / "Time-Series-Library",
        branch="itransformer-weather-eval",
        commit="493256f1584e126ca2d4c3cf93d0a4a339791153",
    )

    # Remove all .json and .csv files from the cloned repository
    # (the path previously used lowercase "time-series-library", so nothing was removed)
    for file in (public / "Time-Series-Library").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()

    # move weather data
    weather_data = raw / "weather.csv"
    weather_data.rename(public / "weather.csv")

    # the description points the agent to the excerpt of the iTransformer paper
    shutil.copy(raw / "itransformer.txt", public / "itransformer.txt")
