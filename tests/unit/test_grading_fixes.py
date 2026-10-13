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


def test_osic_leaderboard_lists_every_kaggle_team(tmp_path):
    from timeseriesgym.registry import Registry

    competition = Registry(tmp_path).get_competition("osic-pulmonary-fibrosis-progression")
    leaderboard = get_leaderboard(competition)
    # it held the top 20 of Kaggle's 2,097 teams, so gold was rank 2 (-6.8311), not Kaggle's
    # rank 10 + int(0.002 * 2097) = 14
    assert len(leaderboard) == 2097 and leaderboard["teamId"].is_unique
    info = competition.grader.rank_score(-6.8412, leaderboard)
    assert info["gold_threshold"] == -6.8412 and info["gold_medal"]
    assert info["silver_threshold"] == -6.8533 and info["bronze_threshold"] == -6.8683
    assert not competition.grader.rank_score(-6.8416, leaderboard)["gold_medal"]  # rank 15


COMPETITIONS = Path(__file__).resolve().parents[2] / "timeseriesgym" / "competitions"
# The columns of Kaggle's leaderboard view, which lists a finished competition's final standings,
# as `ensure_leaderboard_exists` writes them; m5's file keeps the team names. Kaggle's leaderboard
# download (`Rank,TeamId,TeamName,LastSubmissionDate,...`) is the public leaderboard: the ashrae,
# online-sales and sberbank files were that, and recruit's another competition's.
LEADERBOARD_COLUMNS = (
    ["scoreNullable", "teamId", "hasTeamName", "submissionDate", "score", "hasScore"],
    ["teamId", "teamName", "submissionDate", "score"],
)
# The last date in the public data of the challenges whose data have dates (file, column): the
# competition's leaderboard can only be later. recruit's old file was dated 2016-05-24 to
# 2016-08-30.
DATA_ENDS = {
    "ashrae-energy-prediction": "2016-12-31",  # test_features.csv, timestamp
    "child-mind-institute-detect-sleep-states": "2019-07-05",  # train_events.csv, timestamp
    "m5-forecasting-accuracy": "2016-06-19",  # calendar.csv, date
    "new-york-city-taxi-fare-prediction": "2015-06-30",  # test.csv, pickup_datetime
    "recruit-restaurant-visitor-forecasting": "2017-05-31",  # date_info.csv, calendar_date
    "sberbank-russian-housing-market": "2016-10-19",  # macro.csv, timestamp
}
# Challenges taken from a benchmark, not a Kaggle competition: their leaderboard is built from the
# benchmark's published results by a module that pins the source's revision (`REVISION`, of
# `REVISION_DATE`), and the description names that revision.
BENCHMARK_LEADERBOARDS = {"gift-eval-short-horizon": "leaderboard"}


@pytest.mark.parametrize(
    "competition_id", sorted(p.parent.name for p in COMPETITIONS.glob("*/leaderboard.csv"))
)
def test_shipped_leaderboards_are_final_standings_of_their_competition(competition_id):
    leaderboard = pd.read_csv(COMPETITIONS / competition_id / "leaderboard.csv")
    if len(leaderboard) < 2 and competition_id not in BENCHMARK_LEADERBOARDS:
        return  # a placeholder, which get_leaderboard ignores
    assert list(leaderboard.columns) in LEADERBOARD_COLUMNS
    assert leaderboard["teamId"].is_unique
    scores = leaderboard["score"]  # ranked best first: rank_score reads positions
    assert scores.notna().all()
    assert scores.is_monotonic_increasing or scores.is_monotonic_decreasing
    if competition_id in DATA_ENDS:
        first_day = pd.to_datetime(leaderboard["submissionDate"]).min().normalize()
        assert first_day > pd.Timestamp(DATA_ENDS[competition_id])
    if competition_id in BENCHMARK_LEADERBOARDS:
        source = importlib.import_module(
            f"timeseriesgym.competitions.{competition_id}.{BENCHMARK_LEADERBOARDS[competition_id]}"
        )
        assert len(leaderboard) >= 2
        assert (leaderboard["submissionDate"] == source.REVISION_DATE).all()  # one snapshot
        assert source.REVISION in (COMPETITIONS / competition_id / "description.md").read_text()


# What each challenge's score rewards, from its description: True where a lower score is better
LOWER_IS_BETTER = {
    "LANL-Earthquake-Prediction": True,  # MAE
    "MIT-BIH-Arrhythmia": False,  # accuracy
    "MIT-BIH-Arrhythmia-Weak-Supervision": False,  # accuracy
    "amp-parkinsons-disease-progression-prediction": True,  # SMAPE+1
    "ashrae-energy-prediction": True,  # RMSLE
    "child-mind-institute-detect-sleep-states": False,  # mean average precision
    "context-is-key-moirai": False,  # 1 if the results match the reference
    "context-is-key-nn5-chronos": False,  # 1 if the results match the reference
    "csdi-implement-model-pm25": True,  # MAE
    "csdi-reproduce-pm25": True,  # MAE
    "g2net-gravitational-wave-detection": False,  # ROC AUC
    "gift-eval-short-horizon": True,  # normalized MASE
    "hms-harmful-brain-activity-classification": True,  # KL divergence
    "m5-forecasting-accuracy": True,  # WRMSSE
    "moment-anomaly-detection-score": False,  # 1 if the scores match the reference
    "new-york-city-taxi-fare-prediction": True,  # RMSE
    "online-sales": True,  # RMSLE
    "optiver-realized-volatility-prediction": True,  # RMSPE
    "optiver-realized-volatility-prediction-hyperparameter-search": False,  # improvement
    "optiver-realized-volatility-prediction-missing": True,  # RMSPE
    "osic-pulmonary-fibrosis-progression": False,  # Laplace log likelihood
    "ptb-xl-classification-challenge": False,  # accuracy
    "ptb-xl-classification-challenge-feature-enhancement": False,  # overall score
    "ptb-xl-classification-challenge-hyperparameter-search": False,  # improvement
    "recruit-restaurant-visitor-forecasting": True,  # RMSLE
    "resnet-tensorflow-to-pytorch": False,  # tests passed
    "sberbank-russian-housing-market": True,  # RMSLE
    "spaceship-titanic": False,  # accuracy
    "stomp-R-to-python": False,  # tests passed
    "time-series-exam-1-1": False,  # accuracy
    "time-series-library-itransformer-weather-forecast": False,  # 1 if pred.npy matches
    "time-series-library-long-horizon-forecast": True,  # MSE
    "ucr-anomaly-detection": False,  # adjusted F1
    "ventilator-pressure-prediction": True,  # MAE
    "ventilator-pressure-prediction-missing": True,  # MAE
}


def test_every_challenge_configures_the_direction_of_its_score(tmp_path):
    from timeseriesgym.registry import Registry

    registry = Registry(tmp_path)
    assert sorted(LOWER_IS_BETTER) == sorted(registry.list_competition_ids())
    for competition_id, lower_is_better in LOWER_IS_BETTER.items():
        competition = registry.get_competition(competition_id)
        # was taken from the leaderboard only: None, reported as False, without a real one
        assert competition.grader.is_lower_better() is lower_is_better, competition_id
        leaderboard = get_leaderboard(competition)
        ranked = not (
            competition.hyperparameter_search_config or competition.feature_enhancement_config
        )  # these scores are not on the (inherited) leaderboard's scale
        if leaderboard is not None and ranked:
            scores = leaderboard["score"]
            assert bool(scores.iloc[0] < scores.iloc[-1]) is lower_is_better, competition_id
            assert competition.grader.is_lower_better(leaderboard) is lower_is_better


def _fake_kaggle_api(rows: int, teams: int):
    """A Kaggle client whose leaderboard view gives `rows` rows for `x`, which has `teams` teams."""
    entries = [
        SimpleNamespace(
            scoreNullable=f"{0.1 + i / 1e4:.4f}",
            teamId=i,
            teamNameNullable=f"t{i}",
            teamName=f"t{i}",
            hasTeamName=True,
            submissionDate="2020-10-06 06:58:33",
            score=f"{0.1 + i / 1e4:.4f}",
            hasScore=True,
        )
        for i in range(rows)
    ]
    listed = [
        SimpleNamespace(ref="https://www.kaggle.com/competitions/x-two", teamCount=5),
        SimpleNamespace(ref="https://www.kaggle.com/competitions/x", teamCount=teams),
    ]
    return SimpleNamespace(
        competition_leaderboard_view=lambda competition: entries if competition == "x" else [],
        competitions_list=lambda search, category: listed,
    )


def test_leaderboard_download_refuses_a_truncated_view(tmp_path, monkeypatch):
    import timeseriesgym.data as data

    monkeypatch.chdir(tmp_path)
    competition = SimpleNamespace(id="x", leaderboard=tmp_path / "leaderboard.csv")
    # one page of the view, as kaggle 1.6.17 reads it: it was written, as OSIC's 20 rows were
    monkeypatch.setattr(data, "authenticate_kaggle_api", lambda: _fake_kaggle_api(20, 2097))
    with pytest.raises(RuntimeError, match="20 rows .* 2097 teams"):
        data.ensure_leaderboard_exists(competition, force=True)
    assert not competition.leaderboard.exists()

    # benchmarks make a final leaderboard a little longer than the team count
    monkeypatch.setattr(data, "authenticate_kaggle_api", lambda: _fake_kaggle_api(2150, 2148))
    data.ensure_leaderboard_exists(competition, force=True)
    written = pd.read_csv(competition.leaderboard)
    assert len(written) == 2150 and list(written.columns) in LEADERBOARD_COLUMNS


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


def test_feature_enhancement_without_flake8_is_a_setup_error(tmp_path, monkeypatch):
    import shutil

    from timeseriesgym.grade import grade_feature_enhancement
    from timeseriesgym.registry import Registry

    cid = "ptb-xl-classification-challenge-feature-enhancement"
    competition = Registry(tmp_path / "data").get_competition(cid)
    example = Path(importlib.import_module("timeseriesgym").__file__).parent / "competitions"
    folder = tmp_path / "submission"
    shutil.copytree(example / cid / "example", folder)
    competition.answers.parent.mkdir(parents=True)
    shutil.copyfile(folder / "submission.csv", competition.answers)

    clean = subprocess.CompletedProcess([], 0, "", "")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: clean)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    report = grade_feature_enhancement(folder, competition)
    assert report.valid_submission and report.score["accuracy"] == pytest.approx(1.0)

    # without flake8 no submission can pass the rubric: grading stops (each run was invalid)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="flake8 is not installed"):
        grade_feature_enhancement(folder, competition)


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
