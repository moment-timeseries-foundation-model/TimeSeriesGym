"""Grader of the CSDI PM2.5 challenges; csdi-implement-model-pm25 uses the same one.

The submission has one imputed PM2.5 value per evaluation target: `id,value` rows with the
sample submission's ids, in its order. The score is CSDI's metric, the mean absolute error in
PM2.5 units over all evaluation targets. Lower is better.
"""

import numpy as np
import pandas as pd

from timeseriesgym.competitions.utils import check_finite
from timeseriesgym.utils import InvalidSubmissionError


def _check_ids(submission_ids: pd.Series, answer_ids: pd.Series) -> None:
    """The answers' ids (the sample submission's), each once and in the same order."""
    duplicated = submission_ids.duplicated()
    if duplicated.any():
        raise InvalidSubmissionError(
            f"Submission has {duplicated.sum()} duplicate ids, e.g. "
            f"{submission_ids[duplicated].iloc[0]!r}"
        )
    missing = int((~answer_ids.isin(submission_ids)).sum())
    unexpected = int((~submission_ids.isin(answer_ids)).sum())
    if missing or unexpected:
        raise InvalidSubmissionError(
            f"Submission ids do not match the sample submission's: {missing} of its "
            f"{len(answer_ids)} ids are missing and {unexpected} other ids are present"
        )
    differs = submission_ids.to_numpy() != answer_ids.to_numpy()
    if differs.any():
        row = int(np.argmax(differs))
        raise InvalidSubmissionError(
            f"Submission rows are not in the sample submission's order: row {row} is "
            f"{submission_ids.iloc[row]!r}, expected {answer_ids.iloc[row]!r}"
        )


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    for column in ["id", "value"]:
        assert column in answers.columns, f"Answers must have a `{column}` column"
    assert not answers["id"].duplicated().any(), "Answers have duplicate ids"
    y_true = answers["value"].to_numpy(dtype=float)
    assert np.isfinite(y_true).all(), "Answers must be finite"

    if not isinstance(submission, pd.DataFrame):
        raise InvalidSubmissionError(f"Submission must be a table, not {type(submission)}")
    for column in ["id", "value"]:
        if column not in submission.columns:
            raise InvalidSubmissionError(f"Submission must have a `{column}` column")

    _check_ids(submission["id"].astype(str), answers["id"].astype(str))

    values = submission["value"]
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        raise InvalidSubmissionError("Submission `value` values must be numbers")
    check_finite(values, "value")

    return float(np.mean(np.abs(values.to_numpy(dtype=float) - y_true)))
