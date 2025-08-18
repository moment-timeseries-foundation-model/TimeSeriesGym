import os
from pathlib import Path

import numpy as np


def prepare(raw: Path, public: Path, private: Path):
    data = raw / "data.npy"
    data.rename(public / "data.npy")

    pred = raw / "score.npy"
    pred_array = np.load(pred)
    sample_submission = np.zeros(pred_array.shape)

    pred.rename(private / "test.npy")
    np.save(public / "sample_submission.npy", sample_submission)

    # Download the tsl GitHub repository
    os.system(
        f"cd {public}; git clone https://github.com/moment-timeseries-foundation-model/moment.git"
    )

    # Remove all .json and .csv files from the cloned repository
    for file in (public / "moment").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()
