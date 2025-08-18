import pandas as pd
from sklearn.metrics import roc_auc_score

from timeseriesgym.competitions.utils import prepare_for_auroc_metric
from timeseriesgym.utils import InvalidSubmissionError


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")

    classification_inputs = prepare_for_auroc_metric(
        submission=submission, answers=answers, target_column="target", id_column="id"
    )
    return roc_auc_score(**classification_inputs)
