"""Hyperparameter-search score: the relative improvement over the sample submission.

The baseline files are written by the agent, so they only decide whether a submission is valid.
Measuring the improvement against the agent's own baseline paid more for a worse baseline.
"""

import importlib
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn
from xgboost import XGBRegressor

from timeseriesgym.competitions.utils import has_default_hyperparameters
from timeseriesgym.grade import grade_hyperparameter_search
from timeseriesgym.registry import Registry
from timeseriesgym.utils import InvalidSubmissionError

OPTIVER = "optiver-realized-volatility-prediction-hyperparameter-search"
PTB_XL = "ptb-xl-classification-challenge-hyperparameter-search"
SEARCH = "import optuna\nstudy = optuna.create_study()\nstudy.optimize(objective, n_trials=20)\n"

# Optiver: the sample predicts 0.0, so its RMSPE is 1 and the score is 1 - RMSPE.
OPTIVER_ANSWERS = pd.DataFrame(
    {"row_id": ["0-5", "0-11", "1-5", "1-11"], "target": [0.002, 0.001, 0.004, 0.003]}
)
OPTIVER_IMPROVED = [0.0025, 0.001, 0.003, 0.003]  # RMSPE sqrt(0.03125) = 0.17678
OPTIVER_HONEST_BASELINE = [0.003, 0.002, 0.002, 0.002]  # RMSPE 0.63465
OPTIVER_SCORE = 1 - np.sqrt(0.03125)

# PTB-XL: the sample predicts class 0, so its accuracy is 0.2 here.
PTB_XL_ANSWERS = pd.DataFrame({"label": [0, 1, 2, 3, 4, 1, 2, 3, 0, 1]})
PTB_XL_IMPROVED = [0, 1, 2, 3, 4, 1, 2, 0, 1, 2]  # accuracy 0.7
PTB_XL_HONEST_BASELINE = [0, 1, 2, 3, 0, 0, 0, 0, 0, 0]  # accuracy 0.5
PTB_XL_SCORE = (0.7 - 0.2) / 0.2


def _module(competition_id: str):
    return importlib.import_module(f"timeseriesgym.competitions.{competition_id}.grade")


def _config(competition_id: str) -> dict:
    return Registry().get_competition(competition_id).hyperparameter_search_config


def _torchscript(model: nn.Module, path: Path) -> None:
    torch.jit.script(model).save(str(path))


def _small_net() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(12, 5))


def _bigger_net() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 5))


def _submission(tmp_path, competition_id, baseline_predictions, baseline_model=None):
    """A submission folder, the solution script, and the sample submission's path."""
    folder = tmp_path / "submission"
    folder.mkdir(parents=True, exist_ok=True)
    config = _config(competition_id)
    if competition_id == OPTIVER:
        answers, column, improved = OPTIVER_ANSWERS, "target", OPTIVER_IMPROVED
        baseline_model = XGBRegressor() if baseline_model is None else baseline_model
        (folder / config["baseline_model_name"]).write_bytes(pickle.dumps(baseline_model))
        improved_model = XGBRegressor(max_depth=4, learning_rate=0.1)
        (folder / config["improved_model_name"]).write_bytes(pickle.dumps(improved_model))
        sample = answers.assign(target=0.0)
    else:
        answers, column, improved = PTB_XL_ANSWERS, "label", PTB_XL_IMPROVED
        baseline_model = _small_net() if baseline_model is None else baseline_model
        _torchscript(baseline_model, folder / config["baseline_model_name"])
        _torchscript(_bigger_net(), folder / config["improved_model_name"])
        sample = answers.assign(label=0)
    answers.assign(**{column: baseline_predictions}).to_csv(
        folder / config["baseline_submission_name"], index=False
    )
    answers.assign(**{column: improved}).to_csv(
        folder / config["improved_submission_name"], index=False
    )
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    solution = tmp_path / "search.py"
    solution.write_text(SEARCH)
    return folder, answers, config, solution, tmp_path / "sample_submission.csv"


def _grade(competition_id, folder, answers, config, solution, sample):
    return _module(competition_id).grade_hyperparameter_search(
        folder, answers, config, solution, sample
    )


CASES = {
    # competition: (honest baseline, inflated baseline, expected score)
    OPTIVER: (OPTIVER_HONEST_BASELINE, [100 * y for y in OPTIVER_ANSWERS.target], OPTIVER_SCORE),
    PTB_XL: (PTB_XL_HONEST_BASELINE, [4] * 10, PTB_XL_SCORE),  # class 4: accuracy 0.1
}


# ---------------------------------------------------------------- the score


@pytest.mark.parametrize("competition_id", CASES)
def test_an_inflated_baseline_does_not_raise_the_score(tmp_path, competition_id):
    honest, inflated, expected = CASES[competition_id]
    grade = _module(competition_id).grade
    scores, old_scores = [], []
    for name, baseline in [("honest", honest), ("inflated", inflated)]:
        inputs = _submission(tmp_path / name, competition_id, baseline)
        folder, answers, config = inputs[:3]
        scores.append(_grade(competition_id, *inputs))
        # the old score: the improvement over the agent's own baseline
        base = grade(pd.read_csv(folder / config["baseline_submission_name"]), answers)
        imp = grade(pd.read_csv(folder / config["improved_submission_name"]), answers)
        old_scores.append((base - imp) / base if competition_id == OPTIVER else (imp - base) / base)
    assert scores == pytest.approx([expected, expected])
    assert old_scores[1] > old_scores[0]  # Optiver 0.722 -> 0.998, PTB-XL 0.4 -> 6.0


@pytest.mark.parametrize("competition_id", CASES)
def test_an_honest_submission_is_scored_against_the_sample(tmp_path, competition_id):
    honest, _, expected = CASES[competition_id]
    assert _grade(competition_id, *_submission(tmp_path, competition_id, honest)) == (
        pytest.approx(expected)
    )


@pytest.mark.parametrize("competition_id", CASES)
def test_the_cli_reads_the_challenges_sample_submission(tmp_path, competition_id):
    honest, inflated, expected = CASES[competition_id]
    folder, answers, _, solution, sample = _submission(tmp_path, competition_id, honest)
    competition = Registry(tmp_path / "data").get_competition(competition_id)
    competition.answers.parent.mkdir(parents=True)
    answers.to_csv(competition.answers, index=False)
    competition.sample_submission.parent.mkdir(parents=True)
    shutil.copy(sample, competition.sample_submission)

    report = grade_hyperparameter_search(folder, competition, str(solution))
    assert report.valid_submission
    assert report.score == pytest.approx(round(expected, 5))

    _submission(tmp_path, competition_id, inflated)  # overwrites the baseline files
    assert grade_hyperparameter_search(folder, competition, str(solution)).score == report.score

    competition.sample_submission.unlink()  # no trusted floor: not graded
    report = grade_hyperparameter_search(folder, competition, str(solution))
    assert not report.valid_submission
    assert report.score is None


# ---------------------------------------------------------------- validity


@pytest.mark.parametrize(
    "baseline_model,valid",
    [
        (XGBRegressor(), True),
        (XGBRegressor(n_jobs=4, nthread=2, verbosity=0, random_state=0), True),
        (XGBRegressor().fit(np.eye(8), np.arange(8.0)), True),  # trained
        (XGBRegressor(max_depth=3), False),
        (XGBRegressor(n_estimators=50, objective="reg:absoluteerror"), False),
        (GradientBoostingRegressor(), False),
    ],
)
def test_optiver_baseline_must_be_the_default_xgbregressor(tmp_path, baseline_model, valid):
    inputs = _submission(tmp_path, OPTIVER, OPTIVER_HONEST_BASELINE, baseline_model)
    expected = OPTIVER_SCORE if valid else -np.inf
    assert _grade(OPTIVER, *inputs) == pytest.approx(expected)


def test_default_hyperparameters():
    name = "xgboost.sklearn.XGBRegressor"
    assert has_default_hyperparameters(XGBRegressor(), name)
    assert has_default_hyperparameters(XGBRegressor(objective="reg:squarederror"), name)
    assert not has_default_hyperparameters(XGBRegressor(missing=0.0), name)  # default: nan
    assert not has_default_hyperparameters(XGBRegressor(foo=1), name)  # unknown keyword
    assert not has_default_hyperparameters(XGBRegressor(), "xgboost.sklearn.XGBClassifier")
    assert not has_default_hyperparameters(object(), "builtins.object")  # no get_params


@pytest.mark.parametrize("competition_id", CASES)
def test_models_with_the_same_hyperparameters_are_invalid(tmp_path, competition_id):
    honest = CASES[competition_id][0]
    same = XGBRegressor(max_depth=4, learning_rate=0.1) if competition_id == OPTIVER else None
    folder, *rest = _submission(tmp_path, competition_id, honest, same)
    if competition_id == PTB_XL:  # the same architecture as the improved model
        _torchscript(_bigger_net(), folder / _config(PTB_XL)["baseline_model_name"])
    assert _grade(competition_id, folder, *rest) == -np.inf


@pytest.mark.parametrize(
    "improved_model,different",
    [
        (XGBRegressor(n_jobs=4), False),  # was valid: n_jobs counted as a hyperparameter
        (XGBRegressor(nthread=2, verbosity=0, random_state=7), False),
        (XGBRegressor(n_jobs=4, max_depth=3), True),
    ],
)
def test_runtime_only_settings_are_not_different_hyperparameters(
    tmp_path, improved_model, different
):
    folder, *rest = _submission(tmp_path, OPTIVER, OPTIVER_HONEST_BASELINE)
    (folder / _config(OPTIVER)["improved_model_name"]).write_bytes(pickle.dumps(improved_model))
    expected = OPTIVER_SCORE if different else -np.inf
    assert _grade(OPTIVER, folder, *rest) == pytest.approx(expected)


@pytest.mark.parametrize("competition_id", CASES)
def test_the_solution_code_must_show_a_search(tmp_path, competition_id):
    folder, answers, config, solution, sample = _submission(
        tmp_path, competition_id, CASES[competition_id][0]
    )
    for code in [
        "model = None  # tuned by hand\n",
        "import optuna\nprint(optuna.__version__)\n",  # was a search: any reference counted
        "import optuna\nstudy = optuna.create_study()\n",  # created, never run
    ]:
        solution.write_text(code)
        assert _grade(competition_id, folder, answers, config, solution, sample) == -np.inf
    with pytest.raises(InvalidSubmissionError, match="solution code is required"):
        _grade(competition_id, folder, answers, config, None, sample)


@pytest.mark.parametrize("competition_id", CASES)
def test_unreadable_files_in_the_code_directory_are_skipped(tmp_path, competition_id):
    folder, answers, config, solution, sample = _submission(
        tmp_path, competition_id, CASES[competition_id][0]
    )
    code = tmp_path / "code"
    code.mkdir()
    (code / "a_notes.py").write_bytes("# r\xe9sum\xe9\n".encode("latin-1"))  # not UTF-8
    (code / "a_link.py").symlink_to(tmp_path / "absent.py")  # dangling
    shutil.copy(solution, code / "search.py")
    # each of the first two stopped the grading ("does not exist", UnicodeDecodeError)
    expected = CASES[competition_id][2]
    assert _grade(competition_id, folder, answers, config, code, sample) == pytest.approx(expected)

    (code / "search.py").unlink()
    (code / "linked.py").symlink_to(solution)  # symlinks under the directory are not followed
    assert _grade(competition_id, folder, answers, config, code, sample) == -np.inf


@pytest.mark.parametrize(
    "competition_id,baseline",
    [
        (OPTIVER, OPTIVER_HONEST_BASELINE[:3]),  # a missing row
        (OPTIVER, [np.nan, *OPTIVER_HONEST_BASELINE[1:]]),
        (PTB_XL, PTB_XL_HONEST_BASELINE[:9]),
    ],
)
def test_the_baseline_submission_must_be_valid(tmp_path, competition_id, baseline):
    folder, answers, config, solution, sample = _submission(
        tmp_path, competition_id, CASES[competition_id][0]
    )
    column = "target" if competition_id == OPTIVER else "label"
    pd.DataFrame({**answers.iloc[: len(baseline)].to_dict("list"), column: baseline}).to_csv(
        folder / config["baseline_submission_name"], index=False
    )
    with pytest.raises((InvalidSubmissionError, ValueError)):
        _grade(competition_id, folder, answers, config, solution, sample)


@pytest.mark.parametrize("competition_id", CASES)
def test_grading_needs_the_sample_submission(tmp_path, competition_id):
    folder, answers, config, solution, sample = _submission(
        tmp_path, competition_id, CASES[competition_id][0]
    )
    for missing in [None, tmp_path / "absent.csv"]:
        with pytest.raises(ValueError, match="sample submission is required"):
            _grade(competition_id, folder, answers, config, solution, missing)
