import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from timeseriesgym.utils import read_csv

from .constants import TARGET_COLS


def prepare(raw: Path, public: Path, private: Path):
    random_state = 0  # Used for reproducibility
    test_size = 0.1  # 10% of the data will be used for testing

    old_train = read_csv(raw / "train.csv")

    # split the data based on `patient_id` which is the coarsest among all ids
    # to avoid data leakage

    train_patients, test_patients = train_test_split(
        old_train["patient_id"].unique(), test_size=test_size, random_state=random_state
    )

    new_train = old_train[old_train["patient_id"].isin(train_patients)]
    new_test = old_train[old_train["patient_id"].isin(test_patients)]
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "test.csv", index=False)

    new_test_without_labels = new_test.copy().drop(columns=["expert_consensus", *TARGET_COLS])
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # The following 3 sets of id (combinations) uniquely identifies a sample in the metadata
    # 1. label_id
    # 2. (eeg_id, eeg_sub_id)
    # 3. (spectrogram_id, spectrogram_sub_id)
    gold_submission = new_test.copy()[["eeg_id", "eeg_sub_id", *TARGET_COLS]]
    # make the votes into probabilities naively
    # https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468705#2606605
    gold_submission[TARGET_COLS] = gold_submission[TARGET_COLS].div(
        gold_submission[TARGET_COLS].sum(axis=1), axis=0
    )
    gold_submission.to_csv(private / "gold_submission.csv", index=False)

    sample_submission = gold_submission.copy()
    sample_submission[TARGET_COLS] = 1 / len(TARGET_COLS)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    shutil.copytree(raw / "example_figures", public / "example_figures")

    (public / "train_eegs").mkdir(parents=True, exist_ok=True)
    for eeg_id in tqdm(
        new_train["eeg_id"].unique(), desc="Train EEGs", total=len(new_train["eeg_id"].unique())
    ):
        shutil.copy(
            raw / "train_eegs" / f"{eeg_id}.parquet", public / "train_eegs" / f"{eeg_id}.parquet"
        )
    (public / "test_eegs").mkdir(parents=True, exist_ok=True)
    for eeg_id in tqdm(
        new_test["eeg_id"].unique(), desc="Test EEGs", total=len(new_test["eeg_id"].unique())
    ):
        shutil.copy(
            raw / "train_eegs" / f"{eeg_id}.parquet", public / "test_eegs" / f"{eeg_id}.parquet"
        )

    (public / "train_spectrograms").mkdir(parents=True, exist_ok=True)
    for spectrogram_id in tqdm(
        new_train["spectrogram_id"].unique(),
        desc="Train Spectrograms",
        total=len(new_train["spectrogram_id"].unique()),
    ):
        shutil.copy(
            raw / "train_spectrograms" / f"{spectrogram_id}.parquet",
            public / "train_spectrograms" / f"{spectrogram_id}.parquet",
        )
    (public / "test_spectrograms").mkdir(parents=True, exist_ok=True)
    for spectrogram_id in tqdm(
        new_test["spectrogram_id"].unique(),
        desc="Test Spectrograms",
        total=len(new_test["spectrogram_id"].unique()),
    ):
        shutil.copy(
            raw / "train_spectrograms" / f"{spectrogram_id}.parquet",
            public / "test_spectrograms" / f"{spectrogram_id}.parquet",
        )

    assert len(list((public / "train_eegs").rglob("*"))) == len(
        new_train["eeg_id"].unique()
    ), "Unexpected number of train EEGs Copied"
    assert len(list((public / "test_eegs").rglob("*"))) == len(
        new_test["eeg_id"].unique()
    ), "Unexpected number of test EEGs Copied"

    assert len(list((public / "train_spectrograms").rglob("*"))) == len(
        new_train["spectrogram_id"].unique()
    ), "Unexpected number of train Spectrograms Copied"
    assert len(list((public / "test_spectrograms").rglob("*"))) == len(
        new_test["spectrogram_id"].unique()
    ), "Unexpected number of test Spectrograms Copied"

    assert set(new_train.patient_id).isdisjoint(
        set(new_test.patient_id)
    ), "Some patient_ids are in both train and test"
    assert set(new_train.eeg_id).isdisjoint(
        set(new_test.eeg_id)
    ), "Some eeg_ids are in both train and test"
    assert set(new_train.spectrogram_id).isdisjoint(
        set(new_test.spectrogram_id)
    ), "Some spectrogram_ids are in both train and test"

    assert (
        new_train.columns.tolist() == new_test.columns.tolist()
    ), "Columns mismatch between public train and private test"
    assert len(new_train.columns) == 15, "Unexpected number of columns in public train"
    assert len(new_test.columns) == 15, "Unexpected number of columns in private test"
    assert len(sample_submission.columns) == 8, "Unexpected number of columns in sample submission"
    assert len(gold_submission.columns) == 8, "Unexpected number of columns in gold submission"
    assert len(new_test_without_labels.columns) == 8, "Unexpected number of columns in private test"

    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Expected train + test length to be equal to original train length"
    assert len(new_test_without_labels) == len(
        new_test
    ), "Length mismatch between public test and private test"
    assert len(sample_submission) == len(
        new_test
    ), "Length mismatch between sample submission and private test"
    assert len(gold_submission) == len(
        new_test
    ), "Length mismatch between gold submission and private test"
