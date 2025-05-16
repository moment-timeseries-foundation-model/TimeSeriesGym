import os
from pathlib import Path

import numpy as np

from timeseriesgym.utils import read_csv


def prepare(raw: Path, public: Path, private: Path) -> None:
    # Download the data files
    os.system(
        f"curl -L -o {public}/ref_data.csv https://raw.githubusercontent.com/moment-timeseries-foundation-model/tsmp/stomp-R/stomp_python_challenge/ref_data.csv"
    )
    os.system(
        f"curl -L -o {public}/query_data.csv https://raw.githubusercontent.com/moment-timeseries-foundation-model/tsmp/stomp-R/stomp_python_challenge/query_data.csv"
    )

    # Convert the data files to .npy format
    ref_data = read_csv(public / "ref_data.csv")
    ref_data = ref_data.to_numpy().squeeze()
    np.save(public / "ref_data.npy", ref_data)
    os.system(f"rm {public}/ref_data.csv")

    query_data = read_csv(public / "query_data.csv")
    query_data = query_data.to_numpy().squeeze()
    np.save(public / "query_data.npy", query_data)
    os.system(f"rm {public}/query_data.csv")

    # Download the answers file
    os.system(
        f"curl -L -o {private}/test_results.csv https://raw.githubusercontent.com/moment-timeseries-foundation-model/tsmp/stomp-R/stomp_python_challenge/test_results.csv"
    )

    # Download the sample submission file
    os.system(
        f"curl -L -o {public}/sample_submission.py https://raw.githubusercontent.com/moment-timeseries-foundation-model/tsmp/stomp-R/stomp_python_challenge/sample_submission.py"
    )

    # Download the STOMP R code files
    os.system(
        f"git clone https://github.com/moment-timeseries-foundation-model/tsmp.git {public}/tsmp"
    )
    os.system(f"mv {public}/tsmp/R {public}/")  # the R code are under the R folder
    os.system(f"rm -rf {public}/tsmp")
