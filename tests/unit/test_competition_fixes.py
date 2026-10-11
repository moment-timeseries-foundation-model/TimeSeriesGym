"""Regression tests for fixes to competition graders and prepare scripts.

Each test shows the fixed case and that a valid, correctly ordered submission (or a normal
prepare) is unaffected.
"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def _module(competition_id: str, name: str = "grade"):
    return importlib.import_module(f"timeseriesgym.competitions.{competition_id}.{name}")


def _grade_fn(competition_id: str):
    return _module(competition_id).grade


# ---------------------------------------------------------------- amp prepare


def _amp_raw(raw: Path) -> None:
    rows = []
    for patient in range(10, 30):
        for month, rated in [(0, 4), (6, 4), (12, 1)]:  # month 12: only part 1 is rated
            ratings = [
                patient * 10 + month + part if part <= rated else np.nan for part in range(1, 5)
            ]
            rows.append([f"{patient}_{month}", patient, month, *ratings, "On"])
    columns = ["visit_id", "patient_id", "visit_month", "updrs_1", "updrs_2", "updrs_3", "updrs_4"]
    clinical = pd.DataFrame(rows, columns=[*columns, "upd23b_clinical_state_on_medication"])
    raw.mkdir(parents=True)
    clinical.to_csv(raw / "train_clinical_data.csv", index=False)
    clinical.head(0).to_csv(raw / "supplemental_clinical_data.csv", index=False)
    pd.DataFrame({"visit_id": ["10_0"], "patient_id": [10], "Peptide": ["P"]}).to_csv(
        raw / "train_peptides.csv", index=False
    )
    pd.DataFrame({"visit_id": ["10_0"], "patient_id": [10], "UniProt": ["U"]}).to_csv(
        raw / "train_proteins.csv", index=False
    )
    (raw / "example_test_files").mkdir()
    pd.DataFrame(columns=["prediction_id", "rating", "group_key"]).to_csv(
        raw / "example_test_files" / "sample_submission.csv", index=False
    )
    pd.DataFrame(
        columns=["visit_id", "visit_month", "patient_id", "updrs_test", "row_id", "group_key"]
    ).to_csv(raw / "example_test_files" / "test.csv", index=False)


def test_amp_prepare_gives_each_prediction_id_its_own_part(tmp_path):
    prepare = _module("amp-parkinsons-disease-progression-prediction", "prepare").prepare
    raw, public, private = tmp_path / "raw", tmp_path / "public", tmp_path / "private"
    _amp_raw(raw)
    public.mkdir()
    private.mkdir()
    prepare(raw, public, private)

    truth = pd.read_csv(private / "ground_truth.csv")
    sample = pd.read_csv(public / "sample_submission.csv")
    assert truth.prediction_id.is_unique  # was up to 4 rows (one per part) per id
    assert sample.prediction_id.tolist() == truth.prediction_id.tolist()
    clinical = pd.read_csv(raw / "train_clinical_data.csv").set_index("visit_id")
    for prediction_id, rating in zip(truth.prediction_id, truth.rating, strict=True):
        patient, month, _, part, _, plus, _ = prediction_id.split("_")
        assert clinical.loc[f"{patient}_{int(month) + int(plus)}", f"updrs_{part}"] == rating

    patient = truth.prediction_id.str.split("_").str[0].iloc[0]
    ids = set(truth.prediction_id)
    # a target visit with one rated part: the same single answer as before
    assert f"{patient}_0_updrs_1_plus_12_months" in ids
    # no answer for a part that is not rated at the target visit (was given part 1's rating)
    assert f"{patient}_0_updrs_2_plus_12_months" not in ids
    # the public test visits are all test rows with a rating, as before
    test_clinical = pd.read_csv(public / "test_clinical_data.csv")
    assert len(test_clinical) == 9 * test_clinical.patient_id.nunique()
    assert test_clinical.row_id.is_unique
