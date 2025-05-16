from copy import deepcopy
from pathlib import Path

from timeseriesgym.utils import get_logger, read_csv

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path) -> None:
    stop_time = "2016-10-31"  # about 2/12 months of data

    building_metadata = read_csv(raw / "building_metadata.csv")
    building_metadata.to_csv(public / "building_metadata.csv", index=False)

    label = raw / "train.csv"
    features = raw / "weather_train.csv"

    label_df = read_csv(label)
    features_df = read_csv(features)

    train_label_df = label_df[label_df["timestamp"] < stop_time].reset_index(drop=True)
    train_features_df = features_df[features_df["timestamp"] < stop_time].reset_index(drop=True)
    test_label_df = label_df[label_df["timestamp"] >= stop_time].reset_index(drop=True)
    test_features_df = features_df[features_df["timestamp"] >= stop_time].reset_index(drop=True)

    sample_submission = deepcopy(test_label_df)
    sample_submission["meter_reading"] = 0

    train_label_df.to_csv(public / "train_label.csv", index=False)
    train_features_df.to_csv(public / "train_features.csv", index=False)
    test_features_df.to_csv(public / "test_features.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    test_label_df.to_csv(private / "test.csv", index=False)
