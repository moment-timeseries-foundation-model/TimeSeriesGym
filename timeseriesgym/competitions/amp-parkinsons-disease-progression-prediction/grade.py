import pandas as pd

from timeseriesgym.competitions.utils import prepare_for_accuracy_metric
from timeseriesgym.metrics import symmetric_mean_absolute_percentage_error


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    smape_inputs = prepare_for_accuracy_metric(
        submission, answers, target_column="rating", id_column="prediction_id"
    )

    smape = symmetric_mean_absolute_percentage_error(**smape_inputs)

    return smape
