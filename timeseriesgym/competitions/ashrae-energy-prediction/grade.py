import numpy as np
import pandas as pd

from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    submission_array = submission.meter_reading.values
    answers_array = answers.meter_reading.values
    if len(submission_array) != len(answers_array):
        raise InvalidSubmissionError(
            f"Submission length {len(submission_array)} does not match answer "
            "length {len(answers_array)}"
        )

    log_squared_eeror = (np.log(submission_array + 1) - np.log(answers_array + 1)) ** 2
    mean_log_squared_error = np.mean(log_squared_eeror)
    root_mean_log_squared_error = np.sqrt(mean_log_squared_error)

    return root_mean_log_squared_error
