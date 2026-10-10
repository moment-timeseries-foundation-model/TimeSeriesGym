import pandas as pd

from timeseriesgym.metrics import root_mean_squared_log_error
from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission length {len(submission)} does not match answer length {len(answers)}"
        )
    return root_mean_squared_log_error(answers.visitors.to_numpy(), submission.visitors.to_numpy())
