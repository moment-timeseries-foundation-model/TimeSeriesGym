import pandas as pd

from timeseriesgym.competitions.utils import prepare_for_id_matched_metric
from timeseriesgym.metrics import root_mean_squared_log_error


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    inputs = prepare_for_id_matched_metric(
        submission, answers, target_column="visitors", id_columns=["id"]
    )
    return root_mean_squared_log_error(inputs["y_true"], inputs["y_pred"])
