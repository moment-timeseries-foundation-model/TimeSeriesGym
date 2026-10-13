import h5py
import numpy as np
from sklearn.metrics import accuracy_score

from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: h5py.Dataset, answers: h5py.Dataset) -> float:
    # One row per beat and one column per class, exactly as in the answers: argmax over any
    # other number of columns would still give class indices.
    if np.shape(submission) != np.shape(answers):
        raise InvalidSubmissionError(
            f"Submission shape {np.shape(submission)} does not match answers shape "
            f"{np.shape(answers)}"
        )

    y_pred = np.argmax(submission, axis=1)
    y_true = np.argmax(answers, axis=1)

    return accuracy_score(y_true=y_true, y_pred=y_pred)
