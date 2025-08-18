import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if submission.shape != answers.shape:
        raise InvalidSubmissionError(
            f"Shape mismatch: submission {submission.shape} vs answers {answers.shape}."
        )

    submission = submission.sort_values("Id")
    answers = answers.sort_values("Id")

    submission = submission.drop(columns="Id")
    answers = answers.drop(columns="Id")

    y_pred = np.argmax(submission, axis=1)
    y_true = np.argmax(answers, axis=1)

    return accuracy_score(y_true=y_true, y_pred=y_pred)
