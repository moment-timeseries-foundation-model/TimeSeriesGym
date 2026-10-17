from pandas import DataFrame
from sklearn.metrics import root_mean_squared_error

from timeseriesgym.utils import InvalidSubmissionError


def prepare_for_metric(submission: DataFrame, answers: DataFrame) -> dict:
    assert "fare_amount" in answers.columns, "Answers should have a fare_amount column"
    assert "key" in answers.columns, "Answers should have a key column"
    assert answers["key"].is_unique, "Answers should have unique keys"
    if "fare_amount" not in submission.columns:
        raise InvalidSubmissionError("Submission should have a fare_amount column")
    if "key" not in submission.columns:
        raise InvalidSubmissionError("Submission should have a key column")

    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission length {len(submission)} does not match answer length {len(answers)}"
        )

    # Sort by 'key' to ensure alignment
    submission = submission.sort_values("key")
    answers = answers.sort_values("key")

    # The answer keys are unique, so equal sorted keys mean each key is submitted once
    if (submission["key"].values != answers["key"].values).any():
        raise InvalidSubmissionError("Submission and answers `key` values do not match")

    y_true = answers["fare_amount"]
    y_pred = submission["fare_amount"]

    return y_true, y_pred


def grade(submission: DataFrame, answers: DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return root_mean_squared_error(y_true, y_pred)
