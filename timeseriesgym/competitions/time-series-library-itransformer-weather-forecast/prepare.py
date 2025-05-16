import os
from pathlib import Path

import numpy as np


def prepare(raw: Path, public: Path, private: Path):
    solution = raw / "pred.npy"
    solution = np.load(solution)
    sample_submission = np.zeros(solution.shape)

    np.save(private / "test.npy", solution)
    np.save(public / "sample_submission.npy", sample_submission)

    # Download the tsl GitHub repository
    os.system(
        f"cd {public}; git clone --branch itransformer-weather-eval --single-branch https://github.com/raycai420/Time-Series-Library.git"
    )

    # Remove all .json and .csv files from the cloned repository
    for file in (public / "time-series-library").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()
