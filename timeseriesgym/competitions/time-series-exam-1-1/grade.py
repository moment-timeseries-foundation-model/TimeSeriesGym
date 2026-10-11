import pandas as pd
from sklearn.metrics import accuracy_score

from timeseriesgym.competitions.utils import prepare_for_id_matched_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    inputs = prepare_for_id_matched_metric(
        submission, answers, target_column="answer", id_columns=["question_id"]
    )
    return accuracy_score(**inputs)
