from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.special import softmax

from timeseriesgym.utils import InvalidSubmissionError, get_logger

logger = get_logger(__name__)


def df_to_one_hot(
    df: pd.DataFrame, id_column: str, target_column: str, classes: list[str]
) -> pd.DataFrame:
    """
    Convert class labels to one-hot encoded vectors.
    """

    y_onehot = pd.DataFrame(0, index=df.index, columns=[id_column, *classes])
    y_onehot[id_column] = df[id_column]

    for i, row in df.iterrows():
        y_onehot.loc[i, row[target_column]] = 1

    return y_onehot


def one_hot_dfs_to_log_loss_inputs(
    submission_one_hot: pd.DataFrame,
    answers_one_hot: pd.DataFrame,
    id_column: str = "id",
    apply_softmax: bool = True,
) -> dict:
    """
    Frequently used logic to prepare one-hotted dfs for log loss calculation.
    """
    required_cols = set(answers_one_hot.columns)
    submission_cols = set(submission_one_hot.columns)

    if not submission_cols.issuperset(required_cols):
        raise InvalidSubmissionError(
            f"The submission DataFrame is missing some columns required by the `answers` DataFrame."
            f"Missing columns: {required_cols - submission_cols}."
        )
    if id_column not in submission_one_hot.columns:
        raise InvalidSubmissionError(f"Submission is missing id column '{id_column}'.")

    assert id_column in answers_one_hot.columns, f"Answers is missing id column '{id_column}'."

    # Filter submission to only include columns that are in the answers
    submission_filtered = submission_one_hot[
        [col for col in answers_one_hot.columns if col in submission_cols]
    ]

    # Sort submission and answers by id to align them
    submission_sorted = submission_filtered.sort_values(by=id_column).reset_index(drop=True)
    answers_sorted = answers_one_hot.sort_values(by=id_column).reset_index(drop=True)

    assert submission_sorted[id_column].tolist() == answers_sorted[id_column].tolist(), (
        f"Mismatch in {id_column.capitalize()}s between `submission` and `answers` after sorting. "
        f"Number of mismatched {id_column.capitalize()}s: "
        f"{len(set(submission_sorted[id_column]) ^ set(answers_sorted[id_column]))}. "
        f"Ensure both DataFrames have the same {id_column.capitalize()}s."
    )

    assert list(submission_sorted.columns) == list(answers_sorted.columns), (
        "Column order mismatch after filtering and sorting. "
        "Ensure both DataFrames have columns in the same order."
    )

    y_true = answers_sorted.drop(columns=[id_column]).to_numpy()
    y_pred = submission_sorted.drop(columns=[id_column]).to_numpy()

    if apply_softmax and is_one_hot_encoded(y_pred):
        logger.warning(
            "The flag `apply_softmax` has been set to `True` but the submission is already "
            "one-hot encoded. Skipping softmax."
        )

    if apply_softmax and not is_one_hot_encoded(y_pred):
        y_pred = softmax(y_pred, axis=-1)

    log_loss_inputs = {
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return log_loss_inputs


def is_one_hot_encoded(xs: ndarray) -> bool:
    """Check if a 2D NumPy array is one-hot encoded."""

    assert isinstance(xs, ndarray), f"Expected a NumPy array, got {type(xs)}."
    assert xs.ndim == 2, f"Expected a 2D array, got {xs.ndim}D."

    is_binary_matrix = np.bitwise_or(xs == 0, xs == 1).all()
    is_normalized = np.allclose(xs.sum(axis=-1), 1)
    is_one_hot = bool(is_binary_matrix and is_normalized)

    assert isinstance(is_one_hot, bool), f"Expected a boolean, got {type(is_one_hot)}."

    return is_one_hot


def get_ids_from_tf_records(tf_record_path: Path, id_feature: str = "image_name") -> list[str]:
    import tensorflow as tf  # Import only if needed, otherwise it slows down the module import

    tf_record_dataset = tf.data.TFRecordDataset(tf_record_path.as_posix())

    ids = []
    for record in tf_record_dataset:
        features = tf.train.Example.FromString(record.numpy())
        id = features.features.feature[id_feature].bytes_list.value[0].decode("utf-8")
        ids.append(id)

    return ids


def prepare_for_accuracy_metric(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> dict:
    # Answers checks
    assert target_column in answers.columns, f"Answers must have a `{target_column}` column"
    assert id_column in answers.columns, f"Answers must have a `{id_column}` column"

    # Submission checks
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")
    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{target_column}` column")
    if id_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{id_column}` column")

    # sort on id to ensure correct order
    submission = submission.sort_values(id_column)
    answers = answers.sort_values(id_column)

    if (submission[id_column].values != answers[id_column].values).any():
        raise InvalidSubmissionError(f"Submission and Answers `{id_column}`'s do not match")

    y_pred = submission[target_column].to_numpy()
    y_true = answers[target_column].to_numpy()

    return {"y_true": y_true, "y_pred": y_pred}


def prepare_for_auroc_metric(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> dict:
    # Answers checks
    assert id_column in answers.columns, f"answers dataframe should have an {id_column} column"
    assert (
        target_column in answers.columns
    ), f"answers dataframe should have a {target_column} column"

    # Submission checks
    if id_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission should have an {id_column} column")
    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission should have a {target_column} column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers should have the same number of rows")
    try:
        pd.to_numeric(submission[target_column])
    except ValueError as err:
        raise InvalidSubmissionError(
            f"Expected {target_column} to be numeric, got {submission[target_column].dtype} instead"
        ) from err
    if submission[target_column].min() < 0 or submission[target_column].max() > 1:
        raise InvalidSubmissionError(
            f"Submission {target_column} column should contain probabilities,"
            " and therefore contain values between 0 and 1 inclusive"
        )
    # Sort
    submission = submission.sort_values(id_column)
    answers = answers.sort_values(id_column)

    if (submission[id_column].values != answers[id_column].values).any():
        raise InvalidSubmissionError(
            f"Submission and answers should have the same {id_column} values"
        )

    roc_auc_inputs = {
        "y_true": answers[target_column].to_numpy(),
        "y_score": submission[target_column].to_numpy(),
    }

    return roc_auc_inputs
