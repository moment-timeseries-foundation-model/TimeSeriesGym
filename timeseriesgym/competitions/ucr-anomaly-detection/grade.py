import numpy as np

from timeseriesgym.metrics import adjusted_best_f1
from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: dict, answers: dict) -> float:
    """
    Grade the submission by counting the number of matching labels across all datasets.

    Args:
        submission: The submission data dictionary with dataset names as keys
        answers: The answers data dictionary with dataset names as keys

    Returns:
        float: The percentage of matching labels (0.0 to 1.0)
    """
    adjusted_f1_scores = []
    for dataset_name in submission.keys():
        if dataset_name in answers:
            submission_data = submission[dataset_name]
            answers_data = answers[dataset_name]

            if len(submission_data) != len(answers_data):
                raise InvalidSubmissionError(
                    f"Dataset {dataset_name} has different lengths in submission and answers."
                )

            adjusted_f1 = adjusted_best_f1(answers_data, submission_data)
            adjusted_f1_scores.append(adjusted_f1)

    if len(adjusted_f1_scores) == 0:
        raise InvalidSubmissionError("No labels found for evaluation.")

    return np.mean(adjusted_f1_scores)
