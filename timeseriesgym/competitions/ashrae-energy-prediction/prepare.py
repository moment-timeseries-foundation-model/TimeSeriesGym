from copy import deepcopy
from pathlib import Path

from timeseriesgym.utils import read_csv

# The test is the last two months of 2016, and the training labels end two months before it. On
# Kaggle the test (2017-2018) lay up to two years after the training year. With the test right
# after the training labels, each meter's mean log reading over its last 7 training days scored
# better than Kaggle's winner. The weather covers the whole year, as Kaggle's weather files
# covered the test period.
TRAIN_LABELS_END = "2016-09-01"  # training labels before this: 2016-01-01 to 2016-08-31
TEST_START = "2016-10-31"  # test labels and test weather from this on, to 2016-12-31


def prepare(raw: Path, public: Path, private: Path) -> None:
    building_metadata = read_csv(raw / "building_metadata.csv")
    building_metadata.to_csv(public / "building_metadata.csv", index=False)

    label = raw / "train.csv"
    features = raw / "weather_train.csv"

    label_df = read_csv(label)
    features_df = read_csv(features)

    # The readings of 2016-09-01 to 2016-10-30 are in neither split.
    train_label_df = label_df[label_df["timestamp"] < TRAIN_LABELS_END].reset_index(drop=True)
    train_features_df = features_df[features_df["timestamp"] < TEST_START].reset_index(drop=True)
    test_label_df = label_df[label_df["timestamp"] >= TEST_START].reset_index(drop=True)
    test_features_df = features_df[features_df["timestamp"] >= TEST_START].reset_index(drop=True)

    sample_submission = deepcopy(test_label_df)
    sample_submission["meter_reading"] = 0

    train_label_df.to_csv(public / "train_label.csv", index=False)
    train_features_df.to_csv(public / "train_features.csv", index=False)
    test_features_df.to_csv(public / "test_features.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    test_label_df.to_csv(private / "test.csv", index=False)
