"""Regression tests for grading and preparation fixes."""

import importlib
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from timeseriesgym.data import get_leaderboard
from timeseriesgym.grade_helpers import Grader
from timeseriesgym.utils import (
    InvalidSubmissionError,
    PickleModel,
    clone_repository,
    have_different_hyperparameters,
    have_hyperparameter_searching_attempt,
)


def _grade_fn(competition_id: str):
    return importlib.import_module(f"timeseriesgym.competitions.{competition_id}.grade").grade


# ---------------------------------------------------------------- cloned repositories


def test_clone_repository_ships_no_git_history(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(origin), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (origin / "model.py").write_text("SOLUTION = 42\n")
    git("add", ".")
    git("commit", "-q", "-m", "full implementation")
    (origin / "model.py").write_text("SOLUTION = ...  # TODO\n")
    git("commit", "-q", "-am", "remove implementation for the challenge")

    dest = clone_repository(f"file://{origin}", tmp_path / "public" / "repo", branch="main")
    assert (dest / "model.py").read_text() == "SOLUTION = ...  # TODO\n"
    assert not (dest / ".git").exists()
    leftovers = [p for p in dest.rglob("*") if "SOLUTION = 42" in p.read_text(errors="ignore")]
    assert leftovers == []


# ---------------------------------------------------------------- UCR: every dataset is scored


def test_ucr_requires_every_dataset():
    grade = _grade_fn("ucr-anomaly-detection")
    labels = np.zeros(100)
    labels[40:50] = 1
    answers = {"easy": labels, "hard": labels.copy()}
    perfect = {"easy": labels.copy(), "hard": labels.copy()}
    assert grade(perfect, answers) == pytest.approx(1.0)
    with pytest.raises(InvalidSubmissionError, match="missing datasets"):
        grade({"easy": labels.copy()}, answers)


# ---------------------------------------------------------------- exact-match graders


@pytest.mark.parametrize(
    "competition_id",
    ["moment-anomaly-detection-score", "time-series-library-itransformer-weather-forecast"],
)
def test_exact_match_graders_check_shape(competition_id):
    grade = _grade_fn(competition_id)
    answers = np.ones((4, 8))
    assert grade(np.ones((4, 8)), answers) == 1
    assert grade(np.zeros((4, 8)), answers) == 0
    with pytest.raises(InvalidSubmissionError, match="shape"):
        grade(np.ones((1, 8)), answers)  # used to broadcast and pass
    with pytest.raises(InvalidSubmissionError, match="shape"):
        grade(np.float64(1.0), answers)


# ---------------------------------------------------------------- CMI: events without a step


def test_cmi_ignores_ground_truth_events_without_step():
    grade = _grade_fn("child-mind-institute-detect-sleep-states")
    answers = pd.DataFrame(
        {
            "series_id": ["a", "a", "a", "a"],
            "night": [1, 1, 2, 2],
            "event": ["onset", "wakeup", "onset", "wakeup"],
            "step": [100.0, 600.0, np.nan, np.nan],  # night 2 has no events
            "timestamp": ["t"] * 4,
        }
    )
    submission = pd.DataFrame(
        {
            "row_id": [0, 1],
            "series_id": ["a", "a"],
            "step": [100, 600],
            "event": ["onset", "wakeup"],
            "score": [1.0, 1.0],
        }
    )
    assert grade(submission, answers) == pytest.approx(1.0)  # was capped at 0.5


# ---------------------------------------------------------------- leaderboards and medals


def test_placeholder_leaderboard_is_ignored(tmp_path):
    lb = tmp_path / "leaderboard.csv"
    lb.write_text("teamId,teamName,submissionDate,score\n0,dummy,2020-06-30 01:30:59,0\n")
    assert get_leaderboard(SimpleNamespace(id="x", leaderboard=lb)) is None
    lb.write_text(
        "teamId,teamName,submissionDate,score\n0,a,2020,0.9\n1,b,2020,0.8\n2,c,2020,0.7\n"
    )
    assert len(get_leaderboard(SimpleNamespace(id="x", leaderboard=lb))) == 3


def test_non_finite_scores_earn_no_medal():
    grader = Grader(
        name="mae", grade_fn="timeseriesgym.competitions.ventilator-pressure-prediction.grade:grade"
    )
    leaderboard = pd.DataFrame({"score": np.linspace(0.1, 1.0, 50)})  # lower is better
    for score in (float("-inf"), float("inf"), float("nan")):
        info = grader.rank_score(score, leaderboard)
        assert not (info["gold_medal"] or info["silver_medal"] or info["bronze_medal"])
        assert not info["above_median"]
    assert grader.rank_score(0.1, leaderboard)["gold_medal"]


# ---------------------------------------------------------------- hyperparameter search


def _pickled(model, path: Path) -> PickleModel:
    path.write_bytes(pickle.dumps(model))
    return PickleModel.load(str(path))


def test_identical_xgboost_models_have_the_same_hyperparameters(tmp_path):
    xgboost = pytest.importorskip("xgboost")
    a = _pickled(xgboost.XGBRegressor(), tmp_path / "a.pkl")
    b = _pickled(xgboost.XGBRegressor(), tmp_path / "b.pkl")
    assert not have_different_hyperparameters(a, b)  # was True: missing=nan != nan
    c = _pickled(xgboost.XGBRegressor(n_estimators=200, max_depth=4), tmp_path / "c.pkl")
    assert have_different_hyperparameters(a, c)


def test_fitted_state_is_not_a_hyperparameter(tmp_path):
    xgboost = pytest.importorskip("xgboost")
    x, y = np.random.default_rng(0).normal(size=(64, 3)), np.arange(64.0)
    fitted = xgboost.XGBRegressor(n_estimators=5).fit(x, y)
    unfitted = xgboost.XGBRegressor(n_estimators=5)
    a = _pickled(fitted, tmp_path / "a.pkl")
    b = _pickled(unfitted, tmp_path / "b.pkl")
    assert not have_different_hyperparameters(a, b)


@pytest.mark.parametrize(
    "code,expected",
    [
        ("import optuna\nstudy = optuna.create_study()\n", True),
        ("import optuna as opt\nopt.create_study()\n", True),
        ("from optuna import create_study\ncreate_study()\n", True),
        ("from optuna.samplers import TPESampler\ns = TPESampler()\n", True),
        ("import optuna\n", False),  # imported, never used
        ("import numpy as np\nnp.zeros(3)\n", False),
        ("def f(:\n", False),
    ],
)
def test_hyperparameter_search_detection(code, expected):
    assert have_hyperparameter_searching_attempt(code, packages_to_look_for=["optuna"]) is expected
