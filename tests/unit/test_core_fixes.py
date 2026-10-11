"""Regression tests for fixes in the core package, the container environment and the runners."""

from pathlib import Path

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
