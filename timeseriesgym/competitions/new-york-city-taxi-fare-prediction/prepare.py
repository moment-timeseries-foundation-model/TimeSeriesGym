import shutil
from pathlib import Path

from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from timeseriesgym.utils import read_csv

# Kaggle's test set holds only rides that start and end in the NYC area, with 1 to 6
# passengers; train.csv also holds rides with missing (0) or far-away coordinates and with 0 or
# more than 6 passengers. These are the ranges of Kaggle's own test.csv (9,914 rows), pickup and
# dropoff together.
LONGITUDE_RANGE = (-74.263242, -72.986532)
LATITUDE_RANGE = (40.568973, 41.709555)
PASSENGER_COUNT_RANGE = (1, 6)


def like_kaggle_test(df: DataFrame) -> Series:
    """Whether each ride lies within the ranges of Kaggle's test set (bounds included)."""
    keep = df["passenger_count"].between(*PASSENGER_COUNT_RANGE)
    for end in ("pickup", "dropoff"):
        keep &= df[f"{end}_longitude"].between(*LONGITUDE_RANGE)
        keep &= df[f"{end}_latitude"].between(*LATITUDE_RANGE)
    return keep


def prepare(raw: Path, public: Path, private: Path):
    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")

    # Train is c. 55M rows, original test is 9914 rows
    new_train, new_test = train_test_split(old_train, test_size=9914, random_state=0)
    # Clean the test split as Kaggle's was; the training split is not cleaned, and the dropped
    # test rows are used nowhere
    new_test = new_test[like_kaggle_test(new_test)]
    new_test_without_labels = new_test.drop(columns=["fare_amount"])

    # Create a sample submission file
    submission_df = new_test.copy()[["key"]]
    submission_df["fare_amount"] = 11.35

    # Write CSVs
    new_train.to_csv(public / "labels.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    # Copy over other files
    shutil.copy(raw / "GCP-Coupons-Instructions.rtf", public / "GCP-Coupons-Instructions.rtf")

    # Checks
    # build the set from the (small) test split; isdisjoint streams the train keys
    assert set(new_test["key"]).isdisjoint(new_train["key"]), "Train and test sets share samples!"
    assert new_test.shape[1] == 8, f"Test set should have 8 columns, but has {new_test.shape[1]}"
    assert (
        new_test_without_labels.shape[1] == 7
    ), f"Test set without labels should have 7 columns, but has {new_test_without_labels.shape[1]}"
    assert new_train.shape[1] == 8, f"Train set should have 8 columns, but has {new_train.shape[1]}"
    assert (
        submission_df.shape[1] == 2
    ), f"Sample submission should have 2 columns, but has {submission_df.shape[1]}"
    assert (
        submission_df.shape[0] == new_test.shape[0]
    ), f"Sample submission should have {new_test.shape[0]} rows, but has {submission_df.shape[0]}"
