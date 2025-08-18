from copy import deepcopy
from pathlib import Path

from timeseriesgym.utils import get_logger, read_csv

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path) -> None:
    df = read_csv(raw / "TrainingDataset.csv")

    test_frac = 0.10
    seed = 42

    test_df = df.sample(frac=test_frac, random_state=seed)
    train_df = df.drop(test_df.index)

    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    target_columns = [f"Outcome_M{Mi}" for Mi in range(1, 13)]
    test_label_df = test_df[target_columns]
    test_feature_df = test_df.drop(columns=target_columns)

    sample_submission = deepcopy(test_label_df)
    for column in sample_submission.columns:
        sample_submission[column] = 0

    train_df.to_csv(public / "train.csv", index=False)
    test_feature_df.to_csv(public / "test_features.csv", index=False)
    test_label_df.to_csv(private / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
