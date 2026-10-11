from pathlib import Path

import numpy as np
from pandas import DataFrame

from timeseriesgym.competitions.utils import grade_relative_improvement, prepare_for_accuracy_metric


def root_mean_square_percentage_error(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def grade(submission: DataFrame, answers: DataFrame) -> float:
    inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="target", id_column="row_id"
    )
    return root_mean_square_percentage_error(**inputs)


def grade_hyperparameter_search(
    submission_folder_path: Path,
    answers: DataFrame,
    hyperparameter_search_config: dict,
    solution_file_path: Path | None = None,
) -> float:
    return grade_relative_improvement(
        submission_folder_path,
        answers,
        hyperparameter_search_config,
        solution_file_path,
        grade_fn=grade,
        lower_is_better=True,  # RMSPE
        packages_to_look_for=["optuna"],
    )
