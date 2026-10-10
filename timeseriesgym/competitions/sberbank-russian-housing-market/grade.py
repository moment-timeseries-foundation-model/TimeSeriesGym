import pandas as pd

from timeseriesgym.metrics import root_mean_squared_log_error
from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    submission_array = submission.price_doc.values
    answers_array = answers.price_doc.values
    if len(submission_array) != len(answers_array):
        raise InvalidSubmissionError(
            f"Submission length {len(submission_array)} does not match answer length "
            f"{len(answers_array)}"
        )

    return root_mean_squared_log_error(answers_array, submission_array)
