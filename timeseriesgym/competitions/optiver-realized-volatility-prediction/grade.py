import numpy as np
from pandas import DataFrame

from timeseriesgym.competitions.utils import prepare_for_accuracy_metric


def root_mean_square_percentage_error(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def grade(submission: DataFrame, answers: DataFrame) -> float:
    inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="target", id_column="row_id"
    )
    return root_mean_square_percentage_error(**inputs)
