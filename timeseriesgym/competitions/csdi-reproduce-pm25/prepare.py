from pathlib import Path

from timeseriesgym.utils import clone_repository


def prepare(raw: Path, public: Path, private: Path):
    # Download the CSDI GitHub repository
    clone_repository(
        "https://github.com/moment-timeseries-foundation-model/CSDI.git",
        public / "CSDI",
        branch="easy",
    )
