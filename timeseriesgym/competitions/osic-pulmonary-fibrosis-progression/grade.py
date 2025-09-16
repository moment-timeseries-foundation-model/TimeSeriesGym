import numpy as np
from pandas import DataFrame

from timeseriesgym.utils import InvalidSubmissionError


def mean_laplace_log_likelihood(sigma_predicted, fvc_true, fvc_predicted):
    sigma_clipped = np.maximum(sigma_predicted, 70)
    Delta = np.minimum(np.abs(fvc_true - fvc_predicted), 1000)
    metric = -np.sqrt(2) * Delta / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    # averaging the metric across all test set Patient_Week
    return np.mean(metric)


def grade(submission: DataFrame, answers: DataFrame) -> float:
    if "Patient_Week" not in submission.columns:
        raise InvalidSubmissionError("Submission must contain the 'Patient_Week' column")
    if "FVC" not in submission.columns:
        raise InvalidSubmissionError("Submission must contain the 'FVC' column")
    if "Confidence" not in submission.columns:
        raise InvalidSubmissionError("Submission must contain the 'Confidence' column")
    
    submission = submission.loc[submission["Patient_Week"].isin(answers["Patient_Week"])]
    submission = submission.sort_values(by="Patient_Week").reset_index(drop=True)
    answers = answers.sort_values(by="Patient_Week").reset_index(drop=True)
    if len(submission) != len(answers) or (submission["Patient_Week"].values != answers["Patient_Week"].values).any():
        raise InvalidSubmissionError("Submission must include the final three visits for each patient")
    
    return mean_laplace_log_likelihood(
        sigma_predicted=submission["Confidence"].values,
        fvc_true=answers["FVC"].values,
        fvc_predicted=submission["FVC"].values
    )
    

    

    

