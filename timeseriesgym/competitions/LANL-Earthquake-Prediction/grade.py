import pandas as pd
from sklearn.metrics import mean_absolute_error

from timeseriesgym.competitions.utils import prepare_for_accuracy_metric
from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # Prepare for accuracy metric does all the basic checks

    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")

    mae_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="time_to_failure", id_column="seg_id"
    )
    return mean_absolute_error(**mae_inputs)
