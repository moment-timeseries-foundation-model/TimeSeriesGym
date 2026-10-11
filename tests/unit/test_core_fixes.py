"""Regression tests for fixes in the core package, the container environment and the runners."""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from timeseriesgym.utils import (
    InvalidSubmissionError,
    load_models,
    load_scripts,
    load_submissions,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- loaders


@pytest.mark.parametrize("load", [load_submissions, load_scripts, load_models])
def test_loaders_raise_for_missing_files(tmp_path, load):
    # they used to return the exception, which graders then used as data
    with pytest.raises(InvalidSubmissionError, match="does not exist"):
        load(tmp_path / "missing.csv")


class _Payload:
    """Pickles to a call of `Path.touch`: unpickling it creates the marker file."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


def _cik_like_result():
    result = defaultdict(list)
    for task in ("TaskA", "TaskB"):
        for seed in range(1, 4):
            result[task].append({"seed": seed, "score": np.float64(seed / 7)})
    result["arrays"] = [np.arange(3.0), np.ones((2, 2), dtype=np.int32), {1, 2}, (1.5, None)]
    return result


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_pickled_submissions_of_plain_data_load(tmp_path, protocol):
    path = tmp_path / "submission.pkl"
    path.write_bytes(pickle.dumps(_cik_like_result(), protocol=protocol))
    loaded = load_submissions(path)
    assert isinstance(loaded, defaultdict) and loaded["TaskA"] == _cik_like_result()["TaskA"]
    assert np.array_equal(loaded["arrays"][0], np.arange(3.0))


def test_pickled_submissions_cannot_run_code(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "submission.pkl"
    path.write_bytes(pickle.dumps({"TaskA": [_Payload(marker)]}))
    with pytest.raises(InvalidSubmissionError, match="references"):
        load_submissions(path)
    assert not marker.exists()
