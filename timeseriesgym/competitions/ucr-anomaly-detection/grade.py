import numpy as np
from sklearn.metrics import average_precision_score

from timeseriesgym.metrics import adjusted_best_f1
from timeseriesgym.utils import InvalidSubmissionError, get_logger

logger = get_logger(__name__)


def grade(submission: dict, answers: dict) -> float:
    """
    Grade the anomaly scores of each dataset by their average precision on the point labels (the
    area under the precision-recall curve), and average over all datasets.

    Args:
        submission: The submission data dictionary with dataset names as keys
        answers: The answers data dictionary with dataset names as keys

    Returns:
        float: The mean average precision over the datasets (0.0 to 1.0)
    """
    # Average over every dataset in the answers. Iterating over the submission's keys let a
    # submission that contains only the easiest dataset be scored on that dataset alone.
    missing = sorted(set(answers) - set(submission))
    if missing:
        raise InvalidSubmissionError(f"Submission is missing datasets: {missing}")

    average_precisions = []
    adjusted_f1_scores = []
    for dataset_name in answers.keys():
        submission_data = np.asarray(submission[dataset_name])
        answers_data = answers[dataset_name]

        if submission_data.shape != answers_data.shape:
            raise InvalidSubmissionError(
                f"Dataset {dataset_name} has shape {submission_data.shape} in the submission "
                f"and {answers_data.shape} in the answers."
            )
        if submission_data.dtype.kind not in "biuf" or not np.isfinite(submission_data).all():
            raise InvalidSubmissionError(f"Dataset {dataset_name} has non-finite scores.")

        # The score was the point-adjusted best F1, which pays chance: on the shipped data,
        # uniform-random scores earn 0.45 on it (about 0.01 in average precision) and every 400th
        # point set to 1 earns 0.54. It is still logged, for comparison.
        average_precision = average_precision_score(answers_data > 0, submission_data)
        adjusted_f1 = adjusted_best_f1(answers_data, submission_data)
        logger.info(
            f"{dataset_name}: average precision {average_precision:.5f}, "
            f"point-adjusted best F1 {adjusted_f1:.5f} (not scored)"
        )
        average_precisions.append(average_precision)
        adjusted_f1_scores.append(adjusted_f1)

    if len(average_precisions) == 0:
        raise InvalidSubmissionError("No labels found for evaluation.")

    logger.info(f"Mean point-adjusted best F1 (not scored): {np.mean(adjusted_f1_scores):.5f}")
    return float(np.mean(average_precisions))
