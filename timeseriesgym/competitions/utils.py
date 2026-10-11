from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.special import softmax

from timeseriesgym.utils import (
    InvalidSubmissionError,
    get_logger,
    have_different_hyperparameters,
    have_hyperparameter_searching_attempt,
    load_models,
    load_scripts,
    load_submissions,
)

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


def prepare_for_id_matched_metric(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    target_column: str,
    id_columns: list[str],
) -> dict:
    """
    Pairs each answer row with the submission row that has the same ids, in the answers' order,
    so that a correct submission gets the same score in any row order.

    The submission must contain exactly the answers' ids, each once.
    """
    for column in [*id_columns, target_column]:
        assert column in answers.columns, f"Answers must have a `{column}` column"
        if column not in submission.columns:
            raise InvalidSubmissionError(f"Submission must have a `{column}` column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission length {len(submission)} does not match answer length {len(answers)}"
        )

    submission_ids = pd.MultiIndex.from_frame(submission[id_columns])
    if submission_ids.has_duplicates:
        raise InvalidSubmissionError(f"Submission has duplicate {id_columns} values")
    rows = submission_ids.get_indexer(pd.MultiIndex.from_frame(answers[id_columns]))
    if (rows < 0).any():
        raise InvalidSubmissionError(
            f"Submission and answers {id_columns} values do not match: "
            f"{(rows < 0).sum()} answer rows have no submission row"
        )

    y_pred = submission[target_column].to_numpy()[rows]
    y_true = answers[target_column].to_numpy()

    return {"y_true": y_true, "y_pred": y_pred}


def check_finite(values, name: str) -> None:
    """Raises InvalidSubmissionError unless every value is a finite number."""
    try:
        finite = np.isfinite(np.asarray(values, dtype=float)).all()
    except (TypeError, ValueError) as err:
        raise InvalidSubmissionError(f"Submission `{name}` values must be numbers") from err
    if not finite:
        raise InvalidSubmissionError(f"Submission `{name}` values must be finite (no NaN or inf)")


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


def grade_relative_improvement(
    submission_folder_path: Path,
    answers: pd.DataFrame,
    hyperparameter_search_config: dict,
    solution_file_path: Path | None,
    grade_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    lower_is_better: bool,
    packages_to_look_for: list[str] | None = None,
) -> float:
    """
    Grades a hyperparameter search as the relative improvement of the improved submission's
    score over the baseline submission's score, negated when lower scores are better.

    The solution code is required: `solution_file_path` is a script, or a directory whose `.py`
    files are searched. Raises InvalidSubmissionError without it. Returns -inf if the solution
    shows no hyperparameter search with one of `packages_to_look_for` (see
    `have_hyperparameter_searching_attempt` for the default list), or if the two models have
    the same hyperparameters.
    """
    # Checked first, before any submitted model is loaded.
    if solution_file_path is None or not Path(solution_file_path).exists():
        raise InvalidSubmissionError(
            "Submission invalid! The solution code is required to grade a hyperparameter search, "
            f"but none was found at {solution_file_path}."
        )
    solution_path = Path(solution_file_path)
    scripts = sorted(solution_path.rglob("*.py")) if solution_path.is_dir() else [solution_path]
    logger.info("[INFO] Searching the solution code for a hyperparameter searching attempt.")
    if not any(
        have_hyperparameter_searching_attempt(
            load_scripts(script), packages_to_look_for=packages_to_look_for
        )
        for script in scripts
    ):
        logger.info("[INFO] Hyperparameter searching attempt not found.")
        return np.inf * -1

    baseline_model = load_models(
        submission_folder_path / hyperparameter_search_config["baseline_model_name"]
    )
    improved_model = load_models(
        submission_folder_path / hyperparameter_search_config["improved_model_name"]
    )
    baseline_submission = load_submissions(
        submission_folder_path / hyperparameter_search_config["baseline_submission_name"]
    )
    improved_submission = load_submissions(
        submission_folder_path / hyperparameter_search_config["improved_submission_name"]
    )

    if not have_different_hyperparameters(baseline_model, improved_model):
        logger.info("[INFO] The hyperparameters are the same as the baseline model.")
        return np.inf * -1

    baseline_score = grade_fn(baseline_submission, answers)
    improved_score = grade_fn(improved_submission, answers)

    sign = -1 if lower_is_better else 1
    improvement = sign * (improved_score - baseline_score) / baseline_score
    logger.info(f"[INFO] Baseline score: {baseline_score}")
    logger.info(f"[INFO] Improved score: {improved_score}")
    logger.info(f"[INFO] Improvement: {improvement}")

    return improvement
