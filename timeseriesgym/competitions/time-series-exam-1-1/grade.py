import pandas as pd
from sklearn.metrics import accuracy_score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    return accuracy_score(y_true=submission["answer"], y_pred=answers["answer"])
