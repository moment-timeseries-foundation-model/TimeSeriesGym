import h5py
import numpy as np
from sklearn.metrics import accuracy_score

from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: h5py.Dataset, answers: h5py.Dataset) -> float:
    # Prepare for accuracy metric does all the basic checks
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")

    y_pred = np.argmax(submission, axis=1)
    y_true = np.argmax(answers, axis=1)

    return accuracy_score(y_true=y_true, y_pred=y_pred)
