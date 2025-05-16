import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: np.ndarray, answers: np.ndarray) -> float:
    """
    submission: np compound dataset
    answers: np compound dataset
    Returns the
    """
    errors = []
    for dataset_name in answers.dtype.names:
        ground_truth = answers[dataset_name][0]
        prediction = submission[dataset_name][0]

        if ground_truth.shape != prediction.shape:
            raise InvalidSubmissionError(
                f"Shape mismatch: ground truth shape {ground_truth.shape} "
                "!= prediction shape {prediction.shape}"
            )

        if np.isnan(prediction).any():
            raise InvalidSubmissionError(
                f"Prediction contains NaN values: {prediction} for dataset {dataset_name}"
            )

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

    # Calculate the mean of the errors
    mean_error = float(np.mean(errors))
    return mean_error
