import numpy as np

from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: np.array, answers: np.array) -> float:
    # np.isclose broadcasts: without this check a (1, n) or scalar submission could match
    if np.shape(submission) != np.shape(answers):
        raise InvalidSubmissionError(
            f"Submission shape {np.shape(submission)} does not match the expected {np.shape(answers)}"
        )
    close = np.isclose(submission, answers).all()
    return int(close)
