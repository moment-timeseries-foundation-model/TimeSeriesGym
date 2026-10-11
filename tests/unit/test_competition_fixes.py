"""Regression tests for fixes to competition graders and prepare scripts.

Each test shows the fixed case and that a valid, correctly ordered submission (or a normal
prepare) is unaffected.
"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, root_mean_squared_error

from timeseriesgym.metrics import root_mean_squared_log_error
from timeseriesgym.utils import InvalidSubmissionError


def _module(competition_id: str, name: str = "grade"):
    return importlib.import_module(f"timeseriesgym.competitions.{competition_id}.{name}")


def _grade_fn(competition_id: str):
    return _module(competition_id).grade


# ---------------------------------------------------------------- id-matched CSV graders

ID_MATCHED = {
    # competition: (id columns, target column, metric(y_true, y_pred))
    "ashrae-energy-prediction": (
        ["building_id", "meter", "timestamp"],
        "meter_reading",
        root_mean_squared_log_error,
    ),
    "recruit-restaurant-visitor-forecasting": (["id"], "visitors", root_mean_squared_log_error),
    "sberbank-russian-housing-market": (["id"], "price_doc", root_mean_squared_log_error),
    "time-series-exam-1-1": (["question_id"], "answer", accuracy_score),
}


def _id_frame(id_columns, n=12):
    rng = np.random.default_rng(0)
    if id_columns == ["building_id", "meter", "timestamp"]:
        ids = {
            "building_id": np.repeat([0, 1, 2], 4),
            "meter": np.tile([0, 0, 1, 1], 3),
            "timestamp": np.tile(["2016-10-31 00:00:00", "2016-10-31 01:00:00"], 6),
        }
    elif id_columns == ["id"]:
        ids = {"id": [f"air_{i:04d}_2017-03-01" for i in range(n)]}
    else:
        ids = {id_columns[0]: np.arange(1, n + 1)}
    frame = pd.DataFrame(ids)
    return frame, rng


@pytest.mark.parametrize("competition_id", ID_MATCHED)
def test_id_matched_graders_align_rows_by_id(competition_id):
    id_columns, target, metric = ID_MATCHED[competition_id]
    grade = _grade_fn(competition_id)
    answers, rng = _id_frame(id_columns)
    answers[target] = rng.integers(0, 4, len(answers))
    submission = answers.copy()
    submission[target] = rng.integers(0, 4, len(answers))

    # valid, correctly ordered: the positional score, as before
    expected = metric(answers[target].to_numpy(), submission[target].to_numpy())
    assert grade(submission.copy(), answers.copy()) == expected

    # the same predictions in another row order: same score (was scored against other rows)
    shuffled = submission.sample(frac=1, random_state=1).reset_index(drop=True)
    assert not shuffled[id_columns].equals(submission[id_columns])
    assert grade(shuffled, answers.copy()) == pytest.approx(expected)

    # a repeated id in place of another, same length: rejected (was scored)
    repeated = submission.copy()
    repeated.loc[1, id_columns] = repeated.loc[0, id_columns].to_numpy()
    with pytest.raises(InvalidSubmissionError, match="duplicate"):
        grade(repeated, answers.copy())

    # an unknown id: rejected
    unknown = submission.copy()
    column = id_columns[-1]
    unknown[column] = unknown[column].astype(object)
    unknown.loc[0, column] = "unknown"
    with pytest.raises(InvalidSubmissionError, match="do not match"):
        grade(unknown, answers.copy())

    # no id column: rejected (was scored by position)
    with pytest.raises(InvalidSubmissionError, match=f"`{id_columns[0]}`"):
        grade(submission.drop(columns=id_columns[0]), answers.copy())


def test_ashrae_length_message_is_formatted():
    grade = _grade_fn("ashrae-energy-prediction")
    answers, _ = _id_frame(["building_id", "meter", "timestamp"])
    answers["meter_reading"] = 1.0
    with pytest.raises(InvalidSubmissionError, match=r"answer length 12$"):
        grade(answers.iloc[:5].copy(), answers.copy())


def test_nyc_rejects_keys_that_do_not_match():
    grade = _grade_fn("new-york-city-taxi-fare-prediction")
    keys = [f"2010-10-0{i} 21:26:11.0000001" for i in range(1, 7)]
    answers = pd.DataFrame({"key": keys, "fare_amount": [5.0, 8.0, 3.5, 12.0, 7.0, 9.5]})
    submission = pd.DataFrame({"key": keys, "fare_amount": [6.0, 8.0, 4.0, 10.0, 7.5, 9.0]})

    expected = root_mean_squared_error(answers.fare_amount, submission.fare_amount)
    assert grade(submission.copy(), answers.copy()) == pytest.approx(expected)
    shuffled = submission.sample(frac=1, random_state=0)
    assert grade(shuffled, answers.copy()) == pytest.approx(expected)

    reformatted = submission.copy()
    reformatted["key"] = reformatted["key"].str.replace(".0000001", "")  # was scored by position
    with pytest.raises(InvalidSubmissionError, match="do not match"):
        grade(reformatted, answers.copy())
    repeated = submission.copy()
    repeated.loc[1, "key"] = repeated.loc[0, "key"]
    with pytest.raises(InvalidSubmissionError, match="do not match"):
        grade(repeated, answers.copy())
    with pytest.raises(InvalidSubmissionError, match="length"):
        grade(submission.iloc[:-1], answers.copy())


# ---------------------------------------------------------------- MIT-BIH


def test_mit_bih_requires_one_column_per_class():
    grade = _grade_fn("MIT-BIH-Arrhythmia")
    classes = np.array([0, 1, 5, 2, 0, 3, 4, 0])
    answers = np.eye(6)[classes]
    prediction = np.eye(6)[[0, 1, 5, 2, 0, 3, 0, 1]]
    assert grade(prediction, answers) == pytest.approx(
        accuracy_score(classes, [0, 1, 5, 2, 0, 3, 0, 1])
    )
    for shape in [(8, 5), (8, 1), (8, 7), (7, 6)]:
        with pytest.raises(InvalidSubmissionError, match="shape"):
            grade(np.zeros(shape), answers)  # (8, 1) used to score as "class 0 everywhere"


def test_mit_bih_weak_supervision_selects_classes_by_name():
    grade = _grade_fn("MIT-BIH-Arrhythmia-Weak-Supervision")
    pvc = np.array([0, 1, 0, 0, 1, 0])
    answers = pd.DataFrame({"Id": range(6), "normal": 1 - pvc, "pvc": pvc})
    guess = np.array([0, 1, 1, 0, 1, 0])
    submission = pd.DataFrame({"Id": range(6), "normal": 1 - guess, "pvc": guess})
    expected = accuracy_score(pvc, guess)
    assert grade(submission.copy(), answers.copy()) == pytest.approx(expected)

    # the same labels with the class columns in the other order: same score (were all flipped)
    swapped = submission[["Id", "pvc", "normal"]]
    assert grade(swapped, answers.copy()) == pytest.approx(expected)
    shuffled = submission.sample(frac=1, random_state=0)
    assert grade(shuffled, answers.copy()) == pytest.approx(expected)

    with pytest.raises(InvalidSubmissionError, match="missing columns"):
        grade(submission.rename(columns={"pvc": "PVC"}), answers.copy())
    other_ids = submission.assign(Id=range(1, 7))
    with pytest.raises(InvalidSubmissionError, match="do not match"):
        grade(other_ids, answers.copy())


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
