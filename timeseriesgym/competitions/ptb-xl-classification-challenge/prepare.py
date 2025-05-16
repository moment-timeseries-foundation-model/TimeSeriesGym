from pathlib import Path

from timeseriesgym.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # read and send train to public
    train_path = raw / "train_df.h5"
    train_path.rename(public / "train_df.h5")

    # read and send validation to public
    validation_path = raw / "val_df.h5"
    validation_path.rename(public / "val_df.h5")

    # read and send test to public
    test_path = raw / "test_df.h5"
    test_path.rename(public / "test_df.h5")

    # read and send test label to private
    test_label = read_csv(raw / "test_label_df.csv")
    # remove Unnamed: 0 column
    if "Unnamed: 0" in test_label.columns:
        test_label = test_label.drop(columns=["Unnamed: 0"])
    test_label.to_csv(private / "test_label_df.csv", index=False)

    # generate a sample submission file of the same size as the test set
    sample_submission = read_csv(raw / "sample_submission.csv")
    # remove Unnamed: 0 column
    if "Unnamed: 0" in sample_submission.columns:
        sample_submission = sample_submission.drop(columns=["Unnamed: 0"])
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
