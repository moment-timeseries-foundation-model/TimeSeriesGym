import numpy as np
import pandas as pd

from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    target_columns = [f"Outcome_M{Mi}" for Mi in range(1, 13)]

    submission_array = submission[target_columns].values
    answers_array = answers[target_columns].values

    if submission_array.shape != answers_array.shape:
        raise InvalidSubmissionError(
            f"Submission shape {submission_array.shape} does not match answers "
            f"shape {answers_array.shape}"
        )

    log_squared_error = (np.log(submission_array + 1) - np.log(answers_array + 1)) ** 2
    mean_squared_log_error = log_squared_error[~np.isnan(log_squared_error)].mean()

    return np.sqrt(mean_squared_log_error)
