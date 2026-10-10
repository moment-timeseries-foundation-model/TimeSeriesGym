from pathlib import Path

from timeseriesgym.utils import clone_repository


def prepare(raw: Path, public: Path, private: Path):
    # Download the CSDI GitHub repository
    clone_repository(
        "https://github.com/moment-timeseries-foundation-model/CSDI.git",
        public / "CSDI",
        branch="pm25",
        commit="80915dfa3e6a83f0cbd43696bd55124a6257999a",
    )
