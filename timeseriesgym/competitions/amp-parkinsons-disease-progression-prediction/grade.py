import pandas as pd

from timeseriesgym.competitions.utils import check_finite, prepare_for_accuracy_metric
from timeseriesgym.metrics import symmetric_mean_absolute_percentage_error


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    smape_inputs = prepare_for_accuracy_metric(
        submission, answers, target_column="rating", id_column="prediction_id"
    )
    check_finite(smape_inputs["y_pred"], "rating")

    smape = symmetric_mean_absolute_percentage_error(**smape_inputs)

    return smape
