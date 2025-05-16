from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path) -> None:
    # Constants
    random_state = 0  # Used for reproducibility
    test_size = 0.1  # 10% of the data will be used for testing

    peptides = pd.read_csv(raw / "train_peptides.csv")
    proteins = pd.read_csv(raw / "train_proteins.csv")
    clinical_data = pd.read_csv(raw / "train_clinical_data.csv")
    supplemental_clinical_data = pd.read_csv(raw / "supplemental_clinical_data.csv")

    # Divide into train and test sets based on patient IDs
    patient_ids = clinical_data.patient_id.unique()
    train_patient_ids, test_patient_ids = train_test_split(
        patient_ids, test_size=test_size, random_state=random_state, shuffle=True
    )

    train_peptides = peptides[peptides.patient_id.isin(train_patient_ids)]
    test_peptides = peptides[peptides.patient_id.isin(test_patient_ids)]

    train_proteins = proteins[proteins.patient_id.isin(train_patient_ids)]
    test_proteins = proteins[proteins.patient_id.isin(test_patient_ids)]

    train_clinical_data = clinical_data[clinical_data.patient_id.isin(train_patient_ids)]
    test_clinical_data = clinical_data[clinical_data.patient_id.isin(test_patient_ids)]

    # Exclude test patients from supplemental clinical data
    supplemental_clinical_data = supplemental_clinical_data[
        ~supplemental_clinical_data.patient_id.isin(test_patient_ids)
    ]

    # Create the test file
    test = test_clinical_data.drop(columns=["upd23b_clinical_state_on_medication"])
    test = test.melt(
        id_vars=["visit_id", "patient_id", "visit_month"],
        value_vars=["updrs_1", "updrs_2", "updrs_3", "updrs_4"],
        var_name="updrs_type",
        value_name="updrs_value",
    )
    test["group_key"] = test["visit_month"]

    # Rename columns
    test = test.rename(columns={"updrs_value": "rating", "updrs_type": "updrs_test"})
    test["row_id"] = test["visit_id"].astype(str) + "_" + test["updrs_test"].astype(str)
    test = test[
        ["visit_id", "visit_month", "patient_id", "updrs_test", "row_id", "group_key", "rating"]
    ]
    test.sort_values(by=["visit_month", "patient_id", "updrs_test"], inplace=True)
    test.dropna(inplace=True)  # Drop rows with missing ratings

    forecast_months = [0, 6, 12, 24]
    row_id = set(test.row_id)
    prediction_ids = []

    for id in row_id:
        prediction_ids.extend([f"{id}_plus_{f}_months" for f in forecast_months])

    # Create a dataframe for predictions
    predictions = pd.DataFrame(
        {
            "prediction_id": prediction_ids,
        }
    )
    predictions.sort_values(by=["prediction_id"], inplace=True)

    def extract_visit_id(prediction_id):
        # Extract the visit_id from the prediction_id
        parts = prediction_id.split("_")
        patient_id = parts[0]
        visit_month = int(parts[1]) + int(parts[5])  # Extract the visit month
        return f"{patient_id}_{visit_month}"

    predictions["visit_id"] = predictions["prediction_id"].apply(extract_visit_id)
    test_and_submission = test.merge(predictions, on=["visit_id"], how="left")

    # Create the test dataframe
    test_clinical_data = test_and_submission[
        ["visit_id", "visit_month", "patient_id", "updrs_test", "row_id", "group_key"]
    ]
    test_clinical_data = test_clinical_data.drop_duplicates().reset_index(
        drop=True
    )  # Drop duplicates rows

    # Create the submission dataframe
    sample_submission = test_and_submission[["prediction_id", "rating"]]
    sample_submission["group_key"] = [
        int(i.split("_")[1]) for i in sample_submission.prediction_id.tolist()
    ]
    sample_submission["sample_rating"] = [0] * len(sample_submission)  # Initialize with 0 rating
    sample_submission = sample_submission[["prediction_id", "sample_rating", "group_key", "rating"]]
    sample_submission.drop_duplicates(inplace=True)  # Drop duplicates rows
    sample_submission = sample_submission.sort_values(
        by=["prediction_id", "group_key"]
    ).reset_index(drop=True)

    grount_truth = sample_submission[["prediction_id", "rating", "group_key"]].copy()
    sample_submission = sample_submission[["prediction_id", "sample_rating", "group_key"]].copy()
    sample_submission.rename(columns={"sample_rating": "rating"}, inplace=True)

    # Test whether sample submission and test have the same columns
    orig_sample_submission = pd.read_csv(raw / "example_test_files" / "sample_submission.csv")
    orig_test = pd.read_csv(raw / "example_test_files" / "test.csv")

    # Check if the sample submission and original sample submission have the same columns
    assert set(sample_submission.columns) == set(orig_sample_submission.columns)
    assert set(test_clinical_data.columns) == set(orig_test.columns)

    assert len(sample_submission) == len(grount_truth)
    assert set(sample_submission.columns) == set(grount_truth.columns)

    # Save train files
    train_peptides.to_csv(public / "train_peptides.csv", index=False)
    train_proteins.to_csv(public / "train_proteins.csv", index=False)
    train_clinical_data.to_csv(public / "train_clinical_data.csv", index=False)
    supplemental_clinical_data.to_csv(public / "supplemental_clinical_data.csv", index=False)

    # Save test files
    test_peptides.to_csv(public / "test_peptides.csv", index=False)
    test_proteins.to_csv(public / "test_proteins.csv", index=False)
    test_clinical_data.to_csv(public / "test_clinical_data.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Save ground truth
    grount_truth.to_csv(private / "ground_truth.csv", index=False)
