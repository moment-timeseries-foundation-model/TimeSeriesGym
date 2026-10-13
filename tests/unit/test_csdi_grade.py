"""Graders of the CSDI PM2.5 challenges: csdi-reproduce-pm25 and csdi-implement-model-pm25."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timeseriesgym import cli
from timeseriesgym.data import get_leaderboard
from timeseriesgym.grade import validate_submission
from timeseriesgym.registry import Registry
from timeseriesgym.utils import InvalidSubmissionError, load_answers, load_submissions

CSDI = ["csdi-reproduce-pm25", "csdi-implement-model-pm25"]
grade = importlib.import_module("timeseriesgym.competitions.csdi-reproduce-pm25.grade").grade

# `<datetime>_<station>`, the datetime as written in pm25_ground.txt
IDS = [f"2014/03/01 0{hour}:00:00_00100{station}" for hour in (1, 2, 3) for station in (1, 2)]
TRUTH = [138.0, 89.0, 124.0, 85.0, 7.0, 3.0]
CLOSE = [140.0, 89.0, 120.5, 85.0, 7.0, 0.0]  # absolute errors 2, 0, 3.5, 0, 0, 3
CLOSE_MAE = 8.5 / 6
SAMPLE_MAE = sum(TRUTH) / 6  # the sample submission's values are 0.0
OTHER_ID = "2014/03/01 04:00:00_001001"


def _write(path: Path, ids: list, values: list) -> Path:
    pd.DataFrame({"id": ids, "value": values}).to_csv(path, index=False)
    return path


@pytest.fixture
def answers(tmp_path):
    return load_answers(_write(tmp_path / "answers.csv", IDS, TRUTH))


def _grade(tmp_path: Path, answers: pd.DataFrame, ids: list, values: list) -> float:
    return grade(load_submissions(_write(tmp_path / "submission.csv", ids, values)), answers)


def test_both_challenges_use_the_same_grader():
    for competition_id in CSDI:
        module = importlib.import_module(f"timeseriesgym.competitions.{competition_id}.grade")
        assert module.grade is grade


def test_perfect_submission_scores_zero(tmp_path, answers):
    assert _grade(tmp_path, answers, IDS, TRUTH) == 0.0
    assert _grade(tmp_path, answers, IDS, [int(v) for v in TRUTH]) == 0.0  # integers are numbers


def test_score_is_the_mean_absolute_error(tmp_path, answers):
    assert _grade(tmp_path, answers, IDS, CLOSE) == pytest.approx(CLOSE_MAE)
    assert _grade(tmp_path, answers, IDS, [0.0] * 6) == pytest.approx(SAMPLE_MAE)


def _swap(rows: list, i: int, j: int) -> list:
    rows = list(rows)
    rows[i], rows[j] = rows[j], rows[i]
    return rows


@pytest.mark.parametrize(
    "ids, values, message",
    [
        # every value is right for its id, but the rows are not in the sample's order
        (IDS[::-1], TRUTH[::-1], "order"),
        (_swap(IDS, 2, 3), _swap(TRUTH, 2, 3), "row 2 is"),
        (IDS[:-1], TRUTH[:-1], "1 of its 6 ids are missing"),
        ([*IDS, OTHER_ID], [*TRUTH, 1.0], "1 other ids"),
        ([*IDS[:-1], OTHER_ID], TRUTH, "1 of its 6 ids are missing and 1 other"),
        ([*IDS[:-1], IDS[0]], TRUTH, "duplicate"),
        ([*IDS, IDS[0]], [*TRUTH, TRUTH[0]], "duplicate"),
        (IDS, [np.nan, *TRUTH[1:]], "finite"),  # an empty cell
        (IDS, [*TRUTH[:-1], np.inf], "finite"),
        (IDS, [*TRUTH[:-1], -np.inf], "finite"),
        (IDS, [*TRUTH[:-1], "high"], "numbers"),
        (IDS, [True, False] * 3, "numbers"),
    ],
)
def test_invalid_submissions_are_rejected(tmp_path, answers, ids, values, message):
    with pytest.raises(InvalidSubmissionError, match=message):
        _grade(tmp_path, answers, ids, values)


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "`id` column"),  # an empty file
        ("id,value\n", "6 ids are missing"),
        ("id,prediction\n" + "".join(f"{i},1.0\n" for i in IDS), "`value` column"),
        ("value\n" + "1.0\n" * 6, "`id` column"),
    ],
)
def test_submissions_without_the_rows_or_columns_are_rejected(tmp_path, answers, text, message):
    submission = tmp_path / "submission.csv"
    submission.write_text(text)
    with pytest.raises(InvalidSubmissionError, match=message):
        grade(load_submissions(submission), answers)


def test_only_a_table_is_graded(answers):
    with pytest.raises(InvalidSubmissionError, match="table"):
        grade(np.array(TRUTH), answers)  # e.g. a .npy upload


@pytest.mark.parametrize("competition_id", CSDI)
def test_registry_grades_it_as_a_single_unranked_file(competition_id):
    competition = Registry().get_competition(competition_id)
    assert competition.grader.grade_fn is grade
    assert competition.competition_type == "simple" and not competition.awards_medals
    # so grade_jsonl sends it to grade_sample, as the other single-file challenges
    assert competition.hyperparameter_search_config is None
    assert competition.coding_config is None and competition.feature_enhancement_config is None
    assert get_leaderboard(competition) is None  # as upstream: no leaderboard, no medals


@pytest.mark.parametrize("competition_id", CSDI)
def test_grade_cli_grades_a_jsonl_of_submissions(tmp_path, monkeypatch, competition_id):
    data = tmp_path / "data"
    competition = Registry(data).get_competition(competition_id)
    if not competition.answers.suffix == competition.sample_submission.suffix == ".csv":
        pytest.skip("config.yaml does not name the CSV answers and sample submission yet")
    competition.answers.parent.mkdir(parents=True)
    _write(competition.answers, IDS, TRUTH)
    competition.sample_submission.parent.mkdir(parents=True)
    _write(competition.sample_submission, IDS, [0.0] * 6)

    submissions = [
        _write(tmp_path / "perfect.csv", IDS, TRUTH),
        _write(tmp_path / "close.csv", IDS, CLOSE),
        competition.sample_submission,
        _write(tmp_path / "reordered.csv", IDS[::-1], TRUTH[::-1]),
        _write(tmp_path / "nan.csv", IDS, [np.nan, *TRUTH[1:]]),
        tmp_path / "missing.csv",
    ]
    jsonl = tmp_path / "submissions.jsonl"
    entries = [{"competition_id": competition_id, "submission_path": str(p)} for p in submissions]
    jsonl.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    out = tmp_path / "reports"
    args = ["grade", "--submission", str(jsonl), "--output-dir", str(out), "--data-dir", str(data)]
    monkeypatch.setattr(sys, "argv", ["timeseriesgym", *args])
    cli.main()

    (report_file,) = out.glob("*_grading_report.json")
    report = json.loads(report_file.read_text())
    reports = report["competition_reports"]
    scores = [None if r["score"] is None else float(r["score"]) for r in reports]
    assert scores == [0.0, round(CLOSE_MAE, 5), round(SAMPLE_MAE, 5), None, None, None]
    assert [r["valid_submission"] for r in reports] == [True, True, True, False, False, False]
    assert [r["submission_exists"] for r in reports] == [True, True, True, True, True, False]
    assert report["total_medals"] == 0 and report["total_above_median"] == 0
    assert {r["gold_threshold"] for r in reports} == {-1.0}  # null rank info

    # the grading server's check runs the same grader
    assert validate_submission(submissions[0], competition) == (True, "Submission is valid.")
    valid, message = validate_submission(submissions[3], competition)
    assert not valid and "order" in message
