import pandas as pd

from timeseriesgym.metrics import root_mean_squared_log_error
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

    # Missing answers are not scored. Dropping every NaN term, as before, also dropped the
    # cells where a prediction < -1 made the log undefined, so submissions could choose which
    # cells counted.
    return root_mean_squared_log_error(answers_array, submission_array)
