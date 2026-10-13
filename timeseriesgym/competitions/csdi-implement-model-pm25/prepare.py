import importlib
from pathlib import Path

# Both CSDI challenges have the same data; only the branch of the CSDI repository differs.
csdi_pm25 = importlib.import_module("timeseriesgym.competitions.csdi-reproduce-pm25.prepare")


def prepare(raw: Path, public: Path, private: Path):
    csdi_pm25.prepare_csdi(
        raw, public, private, branch="pm25", commit="80915dfa3e6a83f0cbd43696bd55124a6257999a"
    )
