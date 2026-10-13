import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from timeseriesgym.utils import InvalidSubmissionError, get_logger

from .mase import mase, normalized_mase

logger = get_logger(__name__)


def grade(submission: np.ndarray, answers: dict) -> float:
    """
    submission: np compound dataset
    answers: the prepared `labels` group: `target` (the test windows' actual values),
        `seasonal_error` (each window's MASE scale) and `seasonal_naive` (each window's seasonal
        naive forecast), each an np compound dataset with one field per dataset
    Returns GIFT-Eval's normalized MASE: the geometric mean over the datasets of the
    submission's MASE divided by the seasonal naive forecast's MASE. Lower is better; the
    seasonal naive forecast scores 1.0. The previous score, MAPE, is logged.
    """
    if not isinstance(answers, dict):  # answers prepared before the MASE metric
        raise ValueError("The answers lack the MASE scales: prepare gift-eval-short-horizon again")
    targets = answers["target"]
    submitted = getattr(getattr(submission, "dtype", None), "names", None) or ()
    ratios = []
    errors = []
    for dataset_name in targets.dtype.names:
        if dataset_name not in submitted:
            raise InvalidSubmissionError(f"The submission has no field {dataset_name}")
        ground_truth = targets[dataset_name][0]
        prediction = submission[dataset_name][0]

        if ground_truth.shape != prediction.shape:
            raise InvalidSubmissionError(
                f"Shape mismatch: ground truth shape {ground_truth.shape} "
                f"!= prediction shape {prediction.shape}"
            )

        if not np.isfinite(prediction).all():
            raise InvalidSubmissionError(
                f"Prediction contains NaN or infinite values for dataset {dataset_name}"
            )

        scale = answers["seasonal_error"][dataset_name][0]
        submission_mase = mase(ground_truth, prediction, scale)
        seasonal_naive_mase = mase(ground_truth, answers["seasonal_naive"][dataset_name][0], scale)
        if not seasonal_naive_mase > 0:  # the prepared answers are at fault, not the submission
            raise ValueError(f"The seasonal naive MASE of {dataset_name} is {seasonal_naive_mase}")
        ratios.append(submission_mase / seasonal_naive_mase)

        # The previous score, MAPE, is still logged.
        # apply mask to not consider NaN values
        nan_mask = np.isnan(ground_truth)

        prediction = prediction[~nan_mask]
        ground_truth = ground_truth[~nan_mask]

        prediction_length = prediction.shape[-1]
        mape = mean_absolute_percentage_error(
            ground_truth.reshape(-1, prediction_length).T,
            prediction.reshape(-1, prediction_length).T,
            multioutput="raw_values",
        )
        errors.extend(list(mape))
        logger.info(
            f"{dataset_name}: MASE {submission_mase:.6g}, seasonal naive MASE "
            f"{seasonal_naive_mase:.6g}, ratio {ratios[-1]:.6g}, MAPE {mape[0]:.6g}"
        )

    score = normalized_mase(ratios)
    logger.info(f"Normalized MASE {score:.6g}; MAPE (the previous score) {np.mean(errors):.6g}")
    return score
