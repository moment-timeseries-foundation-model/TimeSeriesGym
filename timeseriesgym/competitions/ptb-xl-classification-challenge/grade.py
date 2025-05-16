import pandas as pd
from sklearn.metrics import accuracy_score


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    return accuracy_score(y_true=submission["label"], y_pred=answers["label"])
