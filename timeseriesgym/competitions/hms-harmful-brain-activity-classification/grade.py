import pandas as pd

from timeseriesgym.grade_helpers import InvalidSubmissionError
from timeseriesgym.metrics import kl_divergence_score

from .constants import ID_COLS, TARGET_COLS


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> tuple:
    if not all(col in submission.columns for col in ID_COLS):
        raise InvalidSubmissionError(f"Submission must contain all id colums: {ID_COLS}")
    if not all(col in submission.columns for col in TARGET_COLS):
        raise InvalidSubmissionError(f"Submission must contain all target columns: {TARGET_COLS}")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")
    if not (submission[TARGET_COLS].sum(axis=1).apply(lambda x: round(x, 5) == 1).all()):
        raise InvalidSubmissionError("Submission probabilities must add to 1 for each row")

    assert all(
        col in answers.columns for col in ID_COLS
    ), f"Answers must contain both id colums: {ID_COLS} column"
    assert all(
        col in answers.columns for col in TARGET_COLS
    ), f"Answers must contain all target columns: {TARGET_COLS}"

    submission = submission.sort_values(ID_COLS).reset_index(drop=True)
    answers = answers.sort_values(ID_COLS).reset_index(drop=True)

    answers = answers.copy()[[*ID_COLS, *TARGET_COLS]]
    # normalize answers to be max 1, by taking vote / sum(votes)
    # https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468705#2606605
    answers[TARGET_COLS] = answers[TARGET_COLS].div(answers[TARGET_COLS].sum(axis=1), axis=0)

    answers["row_id"] = answers[ID_COLS].astype(str).agg("-".join, axis=1)
    submission["row_id"] = submission[ID_COLS].astype(str).agg("-".join, axis=1)
    if not set(answers["row_id"]) == set(submission["row_id"]):
        raise InvalidSubmissionError("Submission and answers must have the same IDs")
    assert all(
        answers["row_id"] == submission["row_id"]
    ), "Submission and answers must have the same IDs"

    answers.drop(columns=ID_COLS, inplace=True)
    submission.drop(columns=ID_COLS, inplace=True)

    return submission, answers


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    submission, answers = prepare_for_metric(submission, answers)
    score = kl_divergence_score(answers, submission, "row_id")

    return score
