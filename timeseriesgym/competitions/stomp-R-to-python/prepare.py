import shutil
from pathlib import Path

import numpy as np

from timeseriesgym.competitions.utils import download_file
from timeseriesgym.utils import clone_repository, read_csv


def prepare(raw: Path, public: Path, private: Path) -> None:
    # Download the data files
    download_file(
        "https://raw.githubusercontent.com/moment-timeseries-foundation-model/tsmp/stomp-R/stomp_python_challenge/ref_data.csv",
        public / "ref_data.csv",
    )
    download_file(
        "https://raw.githubusercontent.com/moment-timeseries-foundation-model/tsmp/stomp-R/stomp_python_challenge/query_data.csv",
        public / "query_data.csv",
    )

    # Convert the data files to .npy format
    ref_data = read_csv(public / "ref_data.csv")
    ref_data = ref_data.to_numpy().squeeze()
    np.save(public / "ref_data.npy", ref_data)
    (public / "ref_data.csv").unlink()

    query_data = read_csv(public / "query_data.csv")
    query_data = query_data.to_numpy().squeeze()
    np.save(public / "query_data.npy", query_data)
    (public / "query_data.csv").unlink()

    # Download the answers file
    download_file(
        "https://raw.githubusercontent.com/moment-timeseries-foundation-model/tsmp/stomp-R/stomp_python_challenge/test_results.csv",
        private / "test_results.csv",
    )

    # Download the sample submission file
    download_file(
        "https://raw.githubusercontent.com/moment-timeseries-foundation-model/tsmp/stomp-R/stomp_python_challenge/sample_submission.py",
        public / "sample_submission.py",
    )

    # Download the STOMP R code files
    clone_repository(
        "https://github.com/moment-timeseries-foundation-model/tsmp.git",
        public / "tsmp",
        commit="24c9ff2121436c40b5330c392e072650cf4af1c8",
    )
    shutil.move(public / "tsmp" / "R", public / "R")  # the R code are under the R folder
    shutil.rmtree(public / "tsmp")
