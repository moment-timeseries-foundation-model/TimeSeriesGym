import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from timeseriesgym.utils import InvalidSubmissionError

CLASSES = ["normal", "pvc"]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    assert answers["Id"].is_unique, "Answers should have unique Ids"
    if submission.shape != answers.shape:
        raise InvalidSubmissionError(
            f"Shape mismatch: submission {submission.shape} vs answers {answers.shape}."
        )
    missing = [column for column in ["Id", *CLASSES] if column not in submission.columns]
    if missing:
        raise InvalidSubmissionError(f"Submission is missing columns {missing}.")

    submission = submission.sort_values("Id")
    answers = answers.sort_values("Id")

    # The answer Ids are unique, so equal sorted Ids mean each Id is submitted once
    if (submission["Id"].values != answers["Id"].values).any():
        raise InvalidSubmissionError("Submission and answers `Id` values do not match.")

    # Classes are selected by name: the column order decides the argmax index
    y_pred = np.argmax(submission[CLASSES].to_numpy(), axis=1)
    y_true = np.argmax(answers[CLASSES].to_numpy(), axis=1)

    return accuracy_score(y_true=y_true, y_pred=y_pred)
