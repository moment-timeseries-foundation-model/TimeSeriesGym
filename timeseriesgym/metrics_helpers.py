"""
This script exists to reduce code duplication across metrics.
Source: https://www.kaggle.com/code/metric/kaggle-metric-utilities
Linked from: https://www.kaggle.com/code/metric/kullback-leibler-divergence
Linked from: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification
"""

import numpy as np
import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def treat_as_participant_error(error_message: str, solution: pd.DataFrame | np.ndarray) -> bool:
    """Many metrics can raise more errors than can be handled manually.
    This function attempts to identify errors that can be treated as ParticipantVisibleError
    without leaking any competition data.

    If the solution is purely numeric, and there are no numbers in the error message,
    then the error message is sufficiently unlikely to leak usable data and
    can be shown to participants.

    We expect this filter to reject many safe messages.
    It's intended only to reduce the number of errors we need to manage manually.
    """
    # This check treats bools as numeric
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all(
            pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values
        )
        solution_has_bools = any(pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values)
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if "true" in error_message.lower() or "false" in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    """
    Call score. If that raises an error and that already been specifically handled, just raise it.
    Otherwise make a conservative attempt to identify potential participant visible errors.
    """
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == "ParticipantVisibleError":
            raise ParticipantVisibleError(error_message) from err
        elif err.__class__.__name__ == "HostVisibleError":
            raise HostVisibleError(error_message) from err
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message) from err
            else:
                raise err
    return score_result


def verify_valid_probabilities(df: pd.DataFrame, df_name: str):
    """Verify that the dataframe contains valid probabilities.

    The dataframe must be limited to the target columns; do not pass in any ID columns.
    """
    if not pandas.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(f"All target values in {df_name} must be numeric")

    if df.min().min() < 0:
        raise ParticipantVisibleError(f"All target values in {df_name} must be at least zero")

    if df.max().max() > 1:
        raise ParticipantVisibleError(f"All target values in {df_name} must be no greater than one")

    if not np.allclose(df.sum(axis=1), 1):
        raise ParticipantVisibleError(
            f"Target values in {df_name} do not add to one within all rows"
        )


def adjust_predictions(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Calculate adjusted predictions using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
