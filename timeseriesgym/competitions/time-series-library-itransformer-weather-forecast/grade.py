import numpy as np


def grade(submission: np.array, answers: np.array) -> float:
    close = np.isclose(submission, answers).all()
    return int(close)
