import os
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path):
    # Download the CSDI GitHub repository
    os.system(
        f"cd {public}; git clone --branch pm25 --single-branch https://github.com/moment-timeseries-foundation-model/CSDI.git"
    )
