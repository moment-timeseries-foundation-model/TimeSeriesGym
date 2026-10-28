import pandas as pd

from timeseriesgym.competitions.utils import prepare_for_id_matched_metric
from timeseriesgym.metrics import root_mean_squared_log_error


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    # rows are matched on the (building_id, meter, timestamp) key, not by position
    inputs = prepare_for_id_matched_metric(
        submission,
        answers,
        target_column="meter_reading",
        id_columns=["building_id", "meter", "timestamp"],
    )

    return root_mean_squared_log_error(inputs["y_true"], inputs["y_pred"])
