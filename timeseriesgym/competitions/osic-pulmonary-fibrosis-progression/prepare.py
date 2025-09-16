import pandas as pd
import shutil

from pathlib import Path
from sklearn.model_selection import train_test_split
from timeseriesgym.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    random_state = 0  # Used for reproducibility
    test_size = 0.1  # 10% of the data will be used for testing

    train_patient_ids = sorted(
        [folder.name for folder in (raw / "train").glob("*") if folder.is_dir()]
    )  # all patient ids
    train_df = read_csv(raw / "train.csv")

    assert (
        sorted(train_df["Patient"].unique()) == train_patient_ids
    ), "Expected the patient ids in `train.csv` to match the folders in `train/`."

    new_train_patient_ids, new_test_patient_ids = train_test_split(
        train_patient_ids, test_size=test_size, random_state=random_state
    )

    assert len(set(new_train_patient_ids).intersection(set(new_test_patient_ids))) == 0, (
        f"Expected the new train and test patient ids to be disjoint. Got an intersection of "
        f"{set(new_train_patient_ids).intersection(set(new_test_patient_ids))}."
    )

    new_train_df = train_df[train_df["Patient"].isin(new_train_patient_ids)]
    new_test_df = train_df[train_df["Patient"].isin(new_test_patient_ids)]

    assert len(new_train_df) + len(new_test_df) == len(train_df), (
        f"Expected the number of new train and test instances to sum up to the number of old train "
        f"instances. Got {len(new_train_df)} new train instances and {len(new_test_df)} new test "
        f"instances which sum to {len(new_train_df) + len(new_test_df)} instead of "
        f"{len(train_df)}."
    )

    for new_train_id in new_train_patient_ids:
        shutil.copytree(
            src=raw / "train" / new_train_id,
            dst=public / "train" / new_train_id,
        )

    for new_test_id in new_test_patient_ids:
        shutil.copytree(
            src=raw / "train" / new_test_id,
            dst=public / "test" / new_test_id,
        )
    
    # Construct new train and test ground truth csvs
    new_train_df.to_csv(public / "train.csv", index=False)
    # For the test set, only the initial FVC measurement is provided
    new_test_initial_df = new_test_df.loc[new_test_df.groupby("Patient")["Weeks"].idxmin()]
    new_test_initial_df.to_csv(public / "test.csv", index=False)

    # For test ground truth, only the final three visits are used for grading
    new_test_final_df = new_test_df.groupby("Patient", group_keys=False).apply(
        lambda x: x.nlargest(3, "Weeks").sort_values("Weeks")).reset_index(drop=True)
    new_test_final_df["Patient_Week"] = \
        new_test_final_df["Patient"].astype(str) + "_" + new_test_final_df["Weeks"].astype(str)
    new_test_final_df = new_test_final_df[["Patient_Week", "FVC"]]
    new_test_final_df.to_csv(private / "test.csv", index=False)
    
    # sample submission includes every patient's `FVC` measurement for every possible week
    all_weeks = sorted(
        read_csv(raw / "sample_submission.csv")["Patient_Week"].str.split("_").str[1].unique(),
        key=int
    )
    sample_submission = pd.DataFrame({
        "Patient_Week": [f"{patient}_{week}" for patient in new_test_df["Patient"].unique() for week in all_weeks],
        "FVC": 2000,
        "Confidence": 100,
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
