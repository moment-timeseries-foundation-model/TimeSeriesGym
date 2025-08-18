import numpy as np
import pandas as pd


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    log_squared_error = (
        np.log(submission.visitors.to_numpy() + 1) - np.log(answers.visitors.to_numpy() + 1)
    ) ** 2
    log_squared_error = log_squared_error.mean()

    return np.sqrt(log_squared_error)
