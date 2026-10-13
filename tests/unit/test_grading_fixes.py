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

    # pinned commit: exactly that tree, still without history
    first = subprocess.run(
        ["git", "-C", str(origin), "rev-list", "--max-parents=0", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git("config", "uploadpack.allowAnySHA1InWant", "true")
    pinned = clone_repository(f"file://{origin}", tmp_path / "pinned", commit=first)
    assert (pinned / "model.py").read_text() == "SOLUTION = 42\n"
    assert not (pinned / ".git").exists()


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


def test_ucr_score_is_the_average_precision_that_chance_does_not_earn():
    from sklearn.metrics import average_precision_score

    from timeseriesgym.metrics import adjusted_best_f1

    grade = _grade_fn("ucr-anomaly-detection")
    labels = np.zeros(2000)
    labels[750:850] = 1
    answers = {"a": labels}
    rng = np.random.default_rng(0)
    random = rng.random(2000)
    comb = np.zeros(2000)
    comb[::400] = 1.0  # knows nothing of the labels, but one tooth lands in the anomaly
    for label_free in [random, comb, np.zeros(2000)]:
        assert grade({"a": label_free}, answers) < 0.1  # about the share of anomalous points
    # the old score, the point-adjusted best F1, paid the random scores and the comb
    assert adjusted_best_f1(labels, random) > 0.5
    assert adjusted_best_f1(labels, comb) > 0.9

    honest = labels + rng.normal(0, 0.3, 2000)  # an imperfect detector
    assert grade({"a": honest}, answers) == pytest.approx(average_precision_score(labels, honest))
    assert 0.5 < grade({"a": honest}, answers) < 1.0
    assert grade({"a": labels.copy()}, answers) == 1.0


@pytest.mark.parametrize("bad", ["nan", "inf", "column", "text"])
def test_ucr_scores_must_be_finite_numbers_in_the_shape_of_the_labels(bad):
    grade = _grade_fn("ucr-anomaly-detection")
    labels = np.zeros(100)
    labels[40:50] = 1
    scores = {
        "nan": np.full(100, np.nan),  # was graded 0.0
        "inf": np.r_[np.zeros(99), np.inf],  # was graded 0.0
        "column": labels.reshape(100, 1),  # was graded 1.0
        "text": np.array(["a"] * 100),  # was an unexpected error
    }[bad]
    with pytest.raises(InvalidSubmissionError):
        grade({"a": scores}, {"a": labels})


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
        ("import optuna\nstudy = optuna.create_study()\nstudy.optimize(f)\n", True),
        ("import optuna as opt\nopt.create_study().optimize(f)\n", True),
        ("from optuna import create_study\ncreate_study().optimize(f)\n", True),
        ("import optuna\nstudy = optuna.load_study(name)\ntrial = study.ask()\n", True),
        ("import optuna\nstudy = optuna.create_study()\n", False),  # created, never run
        ("from optuna.samplers import TPESampler\ns = TPESampler()\n", False),
        ("import optuna\nprint(optuna.__version__)\n", False),
        ("import optuna\n", False),  # imported, never used
        ("import numpy as np\nnp.zeros(3)\n", False),
        ("def f(:\n", False),
    ],
)
def test_hyperparameter_search_detection(code, expected):
    assert have_hyperparameter_searching_attempt(code, packages_to_look_for=["optuna"]) is expected


def test_a_listed_library_without_known_search_calls_counts_when_called():
    libraries = ["mylib.tuning"]
    called = "import mylib.tuning as t\nt.search(space)\n"
    assert have_hyperparameter_searching_attempt(called, packages_to_look_for=libraries)
    read = "import mylib.tuning as t\nprint(t.VERSION)\n"
    assert not have_hyperparameter_searching_attempt(read, packages_to_look_for=libraries)


# ---------------------------------------------------------------- infrastructure


def test_container_archives_cannot_escape_on_extraction(tmp_path):
    import io
    import tarfile

    from environment.utils import safe_extractall

    def regular(name: str) -> tuple[tarfile.TarInfo, io.BytesIO]:
        info = tarfile.TarInfo(name)
        info.size = 3
        return info, io.BytesIO(b"1,2")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.addfile(*regular("code/train.py"))
        link = tarfile.TarInfo("submission/submission.csv")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"  # planted by an agent
        tar.addfile(link)
        tar.addfile(*regular("submission/other.csv"))
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r") as tar:
        safe_extractall(tar, tmp_path)
    assert not (tmp_path / "submission" / "submission.csv").exists()
    # the unsafe member is skipped, not the rest of the archive (it used to abort)
    assert (tmp_path / "code" / "train.py").read_bytes() == b"1,2"
    assert (tmp_path / "submission" / "other.csv").read_bytes() == b"1,2"


def test_cli_does_not_import_deepeval():
    code = "import sys, timeseriesgym.cli; print('deepeval' in sys.modules)"
    out = subprocess.run(["python", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_aggregate_reports_accepts_code_reports():
    from timeseriesgym.grade import aggregate_reports

    code_report = SimpleNamespace(submission_exists=True, valid_submission=True, to_dict=lambda: {})
    summary = aggregate_reports([code_report])
    assert summary["total_runs"] == 1 and summary["total_medals"] == 0


# ---------------------------------------------------------------- feature enhancement


def test_feature_enhancement_grader_is_reachable(tmp_path, monkeypatch):
    from timeseriesgym.grade_helpers import GradingInput
    from timeseriesgym.registry import Registry

    cid = "ptb-xl-classification-challenge-feature-enhancement"
    competition = Registry(tmp_path).get_competition(cid)
    clean = subprocess.CompletedProcess([], 0, "", "")  # flake8, which is not installed everywhere
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: clean)
    assert competition.feature_enhancement_config["model_code_file"] == "model.py"
    example = (
        Path(importlib.import_module("timeseriesgym").__file__).parent
        / "competitions"
        / cid
        / "example"
    )
    labels = pd.read_csv(example / "submission.csv")
    score = competition.grader(
        GradingInput(
            submission_folder_path=example,
            answers=labels,  # the example's own predictions: accuracy 1.0
            feature_enhancement_config=competition.feature_enhancement_config,
            solution_files_path=example,
        )
    )
    assert isinstance(score, dict) and score["accuracy"] == pytest.approx(1.0)
    assert not score["rubric_passed"] and score["overall_score"] == 0.0  # no TensorBoard logs


# ---------------------------------------------------------------- RMSLE graders


def test_rmsle_rejects_predictions_that_would_drop_cells():
    from sklearn.metrics import mean_squared_log_error

    from timeseriesgym.metrics import root_mean_squared_log_error

    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.5])
    assert root_mean_squared_log_error(y, p) == pytest.approx(np.sqrt(mean_squared_log_error(y, p)))
    with pytest.raises(InvalidSubmissionError):
        root_mean_squared_log_error(y, np.array([1.0, -5.0, 3.0]))  # used to be dropped
    with pytest.raises(InvalidSubmissionError):
        root_mean_squared_log_error(y, np.array([1.0, np.nan, 3.0]))
    # only cells with a missing *answer* are ignored
    assert root_mean_squared_log_error(np.array([1.0, np.nan]), np.array([1.0, 7.0])) == 0.0


def test_online_sales_cannot_hide_cells():
    grade = _grade_fn("online-sales")
    cols = [f"Outcome_M{i}" for i in range(1, 13)]
    answers = pd.DataFrame([[10.0] * 12, [20.0] * 12], columns=cols)
    good = pd.DataFrame([[10.0] * 12, [10.0] * 12], columns=cols)
    assert grade(good, answers) > 0
    hiding = good.copy()
    hiding.loc[1, cols] = -5.0  # made the badly predicted row vanish from the score
    with pytest.raises(InvalidSubmissionError):
        grade(hiding, answers)
