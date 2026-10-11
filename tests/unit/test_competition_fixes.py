"""Regression tests for fixes to competition graders and prepare scripts.

Each test shows the fixed case and that a valid, correctly ordered submission (or a normal
prepare) is unaffected.
"""

import importlib
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, root_mean_squared_error

from timeseriesgym.grade_helpers import Grader, GradingInput
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


# ---------------------------------------------------------------- non-finite predictions


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_amp_optiver_osic_reject_non_finite_predictions(bad):
    amp = _grade_fn("amp-parkinsons-disease-progression-prediction")
    ids = [f"55_0_updrs_{k}_plus_0_months" for k in range(1, 5)]
    answers = pd.DataFrame({"prediction_id": ids, "rating": [3.0, 0.0, 10.0, 1.0], "group_key": 0})
    submission = answers.assign(rating=[4.0, 0.0, 8.0, 1.0])
    expected = np.mean(200 * np.abs([1.0, 0.0, 2.0, 0.0]) / np.array([9.0, 2.0, 20.0, 4.0]))
    assert amp(submission.copy(), answers.copy()) == pytest.approx(expected)
    with pytest.raises(InvalidSubmissionError, match="finite"):
        amp(submission.assign(rating=[4.0, bad, 8.0, 1.0]), answers.copy())

    optiver = _grade_fn("optiver-realized-volatility-prediction")
    answers = pd.DataFrame({"row_id": ["0-5", "0-11", "1-5"], "target": [0.002, 0.001, 0.004]})
    submission = answers.assign(target=[0.0025, 0.001, 0.003])
    rmspe = np.sqrt(np.mean(((answers.target - submission.target) / answers.target) ** 2))
    assert optiver(submission.copy(), answers.copy()) == pytest.approx(rmspe)
    with pytest.raises(InvalidSubmissionError, match="finite"):
        optiver(submission.assign(target=[0.0025, bad, 0.003]), answers.copy())

    osic = _grade_fn("osic-pulmonary-fibrosis-progression")
    answers = pd.DataFrame({"Patient_Week": ["p_1", "p_2"], "FVC": [3000, 2500]})
    submission = pd.DataFrame(
        {"Patient_Week": ["p_0", "p_1", "p_2"], "FVC": [1, 3100, 2400], "Confidence": [1, 100, 200]}
    )
    sigma, delta = np.array([100, 200]), np.array([100, 100])
    laplace = np.mean(-np.sqrt(2) * delta / sigma - np.log(np.sqrt(2) * sigma))
    assert osic(submission.copy(), answers.copy()) == pytest.approx(laplace)
    for column in ["FVC", "Confidence"]:
        broken = submission.copy()
        broken[column] = broken[column].astype(float)
        broken.loc[1, column] = bad
        with pytest.raises(InvalidSubmissionError, match="finite"):
            osic(broken, answers.copy())  # NaN used to give a "valid" NaN score
    # rows that are not scored may hold anything, as before
    unscored = submission.copy()
    unscored.loc[0, "FVC"] = np.nan
    assert osic(unscored, answers.copy()) == pytest.approx(laplace)


# ---------------------------------------------------------------- M5: ids


def _m5_answers(rng):
    days = [f"d_{i}" for i in range(1, 1942)]
    items = pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D"],
            "dept_id": ["D1", "D1", "D2", "D2"],
            "cat_id": ["C1", "C1", "C2", "C2"],
            "store_id": ["S1", "S1", "S1", "S1"],
            "state_id": ["CA", "CA", "CA", "CA"],
        }
    )
    items.insert(0, "id", [f"{i}_S1_evaluation" for i in items.item_id])
    sales = pd.DataFrame(rng.integers(1, 9, size=(4, len(days))), columns=days)
    test = pd.concat([items, sales], axis=1)
    calendar = pd.DataFrame({"d": [f"d_{i}" for i in range(1, 1970)]})
    calendar["wm_yr_wk"] = 11101 + (np.arange(1969) // 7)
    prices = pd.DataFrame(
        [
            (s, i, w, 1.0 + k)
            for k, i in enumerate("ABCD")
            for s in ["S1"]
            for w in calendar.wm_yr_wk.unique()
        ],
        columns=["store_id", "item_id", "wm_yr_wk", "sell_price"],
    )
    sample = pd.DataFrame({"id": [f"{i}_S1_validation" for i in "ABCD"]})
    for k in range(1, 29):
        sample[f"F{k}"] = 0
    records = []
    for source, frame in [
        ("test", test),
        ("calendar", calendar),
        ("sell_prices", prices),
        ("sample_submission", sample),
    ]:
        records += [dict(r, source=source) for r in frame.to_dict(orient="records")]
    return records, test, sample


def test_m5_rejects_repeated_or_unknown_ids():
    grade = _grade_fn("m5-forecasting-accuracy")
    rng = np.random.default_rng(0)
    answers, test, sample = _m5_answers(rng)
    truth = test[[f"d_{1913 + k}" for k in range(1, 29)]].to_numpy()
    perfect = sample.copy()
    perfect.iloc[:, 1:] = truth
    assert grade(perfect.copy(), answers) == pytest.approx(0.0)

    noisy = sample.copy()
    noisy.iloc[:, 1:] = truth + rng.integers(0, 3, truth.shape)
    score = grade(noisy.copy(), answers)
    assert score > 0
    assert grade(noisy.sample(frac=1, random_state=0), answers) == pytest.approx(score)

    repeated = noisy.copy()
    repeated.loc[1, "id"] = repeated.loc[0, "id"]  # was scored with rows shifted onto other series
    with pytest.raises(InvalidSubmissionError, match="ids"):
        grade(repeated, answers)
    with pytest.raises(InvalidSubmissionError, match="ids"):
        grade(noisy.assign(id=noisy.id.str.replace("_validation", "_evaluation")), answers)


# ---------------------------------------------------------------- hyperparameter search


def _hps_folder(tmp_path):
    from sklearn.linear_model import Ridge

    folder = tmp_path / "submission"
    folder.mkdir()
    (folder / "baseline_model.pkl").write_bytes(pickle.dumps(Ridge(alpha=1.0)))
    (folder / "model.pkl").write_bytes(pickle.dumps(Ridge(alpha=3.0)))
    answers = pd.DataFrame({"row_id": ["0-5", "0-11", "1-5"], "target": [0.002, 0.001, 0.004]})
    answers.assign(target=[0.003, 0.002, 0.002]).to_csv(
        folder / "baseline_submission.csv", index=False
    )
    answers.assign(target=[0.0025, 0.001, 0.003]).to_csv(folder / "submission.csv", index=False)
    return folder, answers


def test_hyperparameter_search_requires_the_solution_code(tmp_path):
    cid = "optiver-realized-volatility-prediction-hyperparameter-search"
    module = _module(cid)
    config = {
        "baseline_submission_name": "baseline_submission.csv",
        "improved_submission_name": "submission.csv",
        "baseline_model_name": "baseline_model.pkl",
        "improved_model_name": "model.pkl",
    }
    folder, answers = _hps_folder(tmp_path)
    baseline = module.grade(pd.read_csv(folder / "baseline_submission.csv"), answers)
    improved = module.grade(pd.read_csv(folder / "submission.csv"), answers)
    expected = -(improved - baseline) / baseline

    script = tmp_path / "search.py"
    script.write_text("import optuna\nstudy = optuna.create_study()\n")
    assert module.grade_hyperparameter_search(folder, answers, config, script) == pytest.approx(
        expected
    )
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "train.py").write_text("print('train')\n")
    shutil.copy(script, code_dir / "search.py")
    assert module.grade_hyperparameter_search(folder, answers, config, code_dir) == pytest.approx(
        expected
    )

    no_search = tmp_path / "manual.py"
    no_search.write_text("from sklearn.linear_model import Ridge\nRidge(alpha=3.0)\n")
    assert module.grade_hyperparameter_search(folder, answers, config, no_search) == -np.inf

    # without the solution code: invalid (was graded), and before any pickle is loaded
    (folder / "model.pkl").write_bytes(b"not a pickle")
    for missing in [None, tmp_path / "absent.py"]:
        with pytest.raises(InvalidSubmissionError, match="solution code is required"):
            module.grade_hyperparameter_search(folder, answers, config, missing)
    grader = Grader(
        name="hps", grade_fn=f"timeseriesgym.competitions.{cid}.grade:grade_hyperparameter_search"
    )
    inputs = GradingInput(
        submission_folder_path=folder, answers=answers, hyperparameter_search_config=config
    )
    assert grader(inputs) is None


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


# ---------------------------------------------------------------- M5 prepare


def test_m5_prepare_keeps_prices_of_every_training_day(tmp_path):
    prepare = _module("m5-forecasting-accuracy", "prepare").prepare
    raw, public, private = tmp_path / "raw", tmp_path / "public", tmp_path / "private"
    for d in (raw, public, private):
        d.mkdir()
    records, test, sample = _m5_answers(np.random.default_rng(0))
    test.to_csv(raw / "sales_train_evaluation.csv", index=False)
    calendar = pd.DataFrame([r for r in records if r["source"] == "calendar"]).drop(
        columns="source"
    )
    calendar.to_csv(raw / "calendar.csv", index=False)
    prices = pd.DataFrame([r for r in records if r["source"] == "sell_prices"]).drop(
        columns="source"
    )
    prices.to_csv(raw / "sell_prices.csv", index=False)
    evaluation = sample.assign(id=sample.id.str.replace("_validation", "_evaluation"))
    pd.concat([sample, evaluation]).to_csv(raw / "sample_submission.csv", index=False)

    prepare(raw, public, private)

    week = calendar.set_index("d")["wm_yr_wk"]
    public_weeks = set(pd.read_csv(public / "sell_prices.csv").wm_yr_wk)
    assert week["d_500"] in public_weeks  # "d_500" > "d_1913" as strings: was dropped
    assert public_weeks == set(week[[f"d_{i}" for i in range(1, 1914)]])
    assert week["d_1941"] not in public_weeks  # forecast-period prices stay out, as before
    train = pd.read_csv(public / "train.csv")
    assert train.columns[-1] == "d_1913" and len(train) == 4
    assert pd.read_csv(public / "sample_submission.csv").id.tolist() == sample.id.tolist()
