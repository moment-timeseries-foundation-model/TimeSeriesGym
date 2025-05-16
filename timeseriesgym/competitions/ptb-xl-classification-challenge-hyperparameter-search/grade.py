from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score

from timeseriesgym.utils import (
    get_logger,
    have_different_hyperparameters,
    have_hyperparameter_searching_attempt,
    load_models,
    load_scripts,
    load_submissions,
)

logger = get_logger(__name__)


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    return accuracy_score(y_true=submission["label"], y_pred=answers["label"])


def grade_hyperparameter_search(
    submission_folder_path: Path,
    answers: DataFrame,
    hyperparameter_search_config: dict,
    solution_file_path: Path | None = None,
) -> float:
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
    has_solution_file = solution_file_path is not None
    solution_file = load_scripts(solution_file_path) if has_solution_file else None

    if not have_different_hyperparameters(baseline_model, improved_model):
        logger.info("[INFO] The hyperparameters are the same as the baseline model.")
        return np.inf * -1

    baseline_score = grade(baseline_submission, answers)
    improved_score = grade(improved_submission, answers)

    improvement = (improved_score - baseline_score) / baseline_score
    logger.info(f"[INFO] Baseline score: {baseline_score}")
    logger.info(f"[INFO] Improved score: {improved_score}")
    logger.info(f"[INFO] Improvement: {improvement}")

    if solution_file is not None:
        logger.info(
            "[INFO] Solution file is provided, searching for hyperparameter searching attempt."
        )
        if not have_hyperparameter_searching_attempt(solution_file):
            logger.info("[INFO] Hyperparameter searching attempt not found.")
            return np.inf * -1

    return improvement
