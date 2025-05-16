from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from timeseriesgym.utils import get_logger, read_csv, read_parquet

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path) -> None:
    random_state = 0
    test_size = 0.1  # 10% of the data will be used for testing

    np.random.seed(random_state)  # Set random seed for reproducibility

    # Read the old training dataset
    logger.info("Reading the training dataset...")

    old_train = read_parquet(raw / "train_series.parquet")
    old_train_events = read_csv(raw / "train_events.csv")

    train_ids, test_ids = train_test_split(
        old_train["series_id"].unique(), test_size=test_size, random_state=random_state
    )

    new_train = old_train[old_train["series_id"].isin(train_ids)]
    new_test = old_train[old_train["series_id"].isin(test_ids)]

    new_train_events = old_train_events[old_train_events["series_id"].isin(train_ids)]
    new_test_events = old_train_events[old_train_events["series_id"].isin(test_ids)]

    # Save the new datasets
    new_train.to_parquet(public / "train_series.parquet", index=False)
    new_test.to_parquet(public / "test_series.parquet", index=False)
    new_train_events.to_csv(public / "train_events.csv", index=False)
    new_test_events.to_csv(private / "test_events.csv", index=False)

    # Sample submission
    test_ids = new_test["series_id"].unique()
    random_onsets = np.random.randint(0, 500, size=len(test_ids))
    random_durations = np.random.randint(0, 50, size=len(test_ids))
    random_wakeups = random_onsets + random_durations

    sample_submission = []
    row_id = 0
    for i, series_id in enumerate(test_ids):
        sample_submission.append(
            {
                "row_id": row_id,
                "series_id": series_id,
                "step": random_onsets[i],
                "event": "onset",
                "score": np.random.rand(),
            }
        )
        row_id += 1
        sample_submission.append(
            {
                "row_id": row_id,
                "series_id": series_id,
                "step": random_wakeups[i],
                "event": "wakeup",
                "score": np.random.rand(),
            }
        )
        row_id += 1

    pd.DataFrame(sample_submission).to_csv(public / "sample_submission.csv", index=False)
    logger.info("New datasets prepared and saved successfully.")
