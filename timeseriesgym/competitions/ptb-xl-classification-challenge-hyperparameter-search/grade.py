from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score

from timeseriesgym.competitions.utils import grade_relative_improvement


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    return accuracy_score(y_true=submission["label"], y_pred=answers["label"])


def grade_hyperparameter_search(
    submission_folder_path: Path,
    answers: DataFrame,
    hyperparameter_search_config: dict,
    solution_file_path: Path | None = None,
    sample_submission_path: Path | None = None,
    private_paths: tuple[Path, ...] | None = None,
) -> float:
    return grade_relative_improvement(
        submission_folder_path,
        answers,
        hyperparameter_search_config,
        solution_file_path,
        sample_submission_path,
        grade_fn=grade,
        private_paths=private_paths,
        lower_is_better=False,  # accuracy
    )
