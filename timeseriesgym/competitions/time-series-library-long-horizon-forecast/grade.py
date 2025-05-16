import numpy as np

from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: dict, answers: dict) -> dict:
    grades = {}

    for ds_name in answers.keys():
        if ds_name not in submission:
            raise InvalidSubmissionError(f"Missing dataset {ds_name} in submission")

        pred = submission[ds_name]
        true = answers[ds_name]
        if pred.shape != true.shape:
            raise InvalidSubmissionError(
                f"Shape mismatch for dataset {ds_name}: {pred.shape} vs {true.shape}"
            )

        mse = np.mean((pred - true) ** 2)
        grades[ds_name] = mse

    return grades
