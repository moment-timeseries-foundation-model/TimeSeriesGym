"""The agent's format check, against a contract derived from the sample submission
(agents/validation.py, environment/validate_submission.py)."""

import io
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from agents.validation import submission_contract
from environment import validate_submission as vs
from timeseriesgym.registry import Registry

REPO = Path(__file__).resolve().parents[2]
HPS_ID = "optiver-realized-volatility-prediction-hyperparameter-search"
FE_ID = "ptb-xl-classification-challenge-feature-enhancement"


def _competition(tmp_path: Path, competition_id: str, sample: bytes | None = None):
    competition = Registry(tmp_path / "data").get_competition(competition_id)
    if sample is not None:
        competition.sample_submission.parent.mkdir(parents=True, exist_ok=True)
        competition.sample_submission.write_bytes(sample)
    return competition


def _contract(tmp_path: Path, competition) -> dict:
    (tmp_path / "submission").mkdir(exist_ok=True)
    return submission_contract(
        competition,
        data_dir=str(competition.public_dir),
        submission_dir=str(tmp_path / "submission"),
        code_dir=str(tmp_path / "code"),
    )


def _validate(tmp_path: Path, contract: dict, path: Path | None = None) -> tuple[int, str]:
    """Runs the agent's command, `validate_submission [path]`; returns its exit code and output."""
    contract_file = tmp_path / "contract.json"
    contract_file.write_text(json.dumps(contract))
    out = io.StringIO()
    argv = [str(path)] if path else []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "stdout", out)
        code = vs.main([*argv, "--contract", str(contract_file)])
    return code, out.getvalue()


# ---------------------------------------------------------------- the format check: CSV

SPACESHIP_SAMPLE = b"PassengerId,Transported\n0001_01,False\n0002_01,False\n0003_01,False\n"


def test_csv_contracts_describe_the_sample(tmp_path):
    competition = _competition(tmp_path, "spaceship-titanic", SPACESHIP_SAMPLE)
    contract = _contract(tmp_path, competition)
    assert contract["kind"] == "file"
    assert contract["file"] == {
        "suffix": ".csv",
        "format": "csv",
        "columns": ["PassengerId", "Transported"],
        "rows": 3,
        "id_columns": ["PassengerId"],  # the column whose values vary in the sample
        "value_columns": {"Transported": {"type": "bool", "missing": False}},
        "sample": "sample_submission.csv",
    }


@pytest.mark.parametrize(
    "content, error",
    [
        (b"PassengerId,Transported\n0001_01,True\n0002_01,False\n0003_01,True\n", None),
        (b"PassengerId,Transported\n0001_01,True\n0002_01,False\n0003_01,1\n", "True or False"),
        (b"PassengerId,Transported\n0001_01,True\n0002_01,False\n", "2 rows"),
        (b"PassengerId,Transported\n0002_01,True\n0001_01,False\n0003_01,True\n", "order"),
        (b"PassengerId,Transported\n0001_01,True\n0002_01,False\n9999_01,True\n", "1 of them"),
        (b"PassengerId,transported\n0001_01,True\n0002_01,False\n0003_01,True\n", "columns"),
        (b"PassengerId,Transported\n0001_01,True\n0002_01,\n0003_01,True\n", "missing values"),
        (b'PassengerId,Transported\n0001_01,True\n"0002_01,False\n', "cannot"),
    ],
)
def test_csv_submissions_must_have_the_format_of_the_sample(tmp_path, content, error):
    contract = _contract(tmp_path, _competition(tmp_path, "spaceship-titanic", SPACESHIP_SAMPLE))
    submission = tmp_path / "submission" / "submission.csv"
    submission.write_bytes(content)
    code, out = _validate(tmp_path, contract, submission)
    if error is None:
        assert code == 0 and out.startswith(vs.VALID), out
        assert "Note" not in out
    else:
        assert code == 1 and out.startswith(vs.INVALID) and error in out, out


@pytest.mark.parametrize("value, valid", [("12.5", True), ("1e400", False), ("abc", False)])
def test_csv_predictions_must_be_finite_numbers(tmp_path, value, valid):
    sample = b"key,fare_amount\na,11.35\nb,11.35\n"
    competition = _competition(tmp_path, "new-york-city-taxi-fare-prediction", sample)
    submission = tmp_path / "submission" / "submission.csv"
    contract = _contract(tmp_path, competition)
    submission.write_text(f"key,fare_amount\na,3\nb,{value}\n")
    assert (_validate(tmp_path, contract)[0] == 0) is valid


def test_detected_events_are_not_held_to_the_rows_of_the_sample(tmp_path):
    sample = b"row_id,series_id,step,event,score\n0,a,10,onset,0.5\n1,a,90,wakeup,0.7\n"
    competition = _competition(tmp_path, "child-mind-institute-detect-sleep-states", sample)
    contract = _contract(tmp_path, competition)  # `submission_format: fixed_rows: false`
    assert contract["file"]["rows"] is None and contract["file"]["id_columns"] == []
    submission = tmp_path / "submission" / "submission.csv"
    submission.write_text("row_id,series_id,step,event,score\n0,b,5,onset,0.9\n")
    assert _validate(tmp_path, contract)[0] == 0
    submission.write_text("row_id,series_id,step,event,score\n0,b,5,onset,high\n")
    assert _validate(tmp_path, contract)[0] == 1


def test_the_contract_holds_nothing_of_the_private_data(tmp_path):
    competition = _competition(tmp_path, "spaceship-titanic", SPACESHIP_SAMPLE)
    assert not competition.private_dir.exists()  # derived from public files only
    contract = _contract(tmp_path, competition)
    competition.private_dir.mkdir(parents=True)
    competition.answers.write_text("PassengerId,Transported\nSECRET_ID,True\n")
    assert _contract(tmp_path, competition) == contract
    assert "SECRET" not in json.dumps(contract)


# ---------------------------------------------------------------- folders and file choice


def test_a_folder_is_checked_by_the_file_that_is_graded(tmp_path):
    contract = _contract(tmp_path, _competition(tmp_path, "spaceship-titanic", SPACESHIP_SAMPLE))
    folder = tmp_path / "submission"
    code, out = _validate(tmp_path, contract)  # default: the submission folder
    assert code == 1 and "has no .csv file" in out
    good, bad = folder / "good.csv", folder / "bad.csv"
    good.write_bytes(SPACESHIP_SAMPLE)
    bad.write_text("PassengerId,Transported\n")
    os.utime(good, (1, 1))  # bad.csv is the last written: make_submission.py grades it
    code, out = _validate(tmp_path, contract)
    assert code == 1 and "0 rows" in out
    code, out = _validate(tmp_path, contract, good)
    assert code == 0 and "bad.csv was written later" in out
    elsewhere = tmp_path / "elsewhere.csv"
    elsewhere.write_bytes(SPACESHIP_SAMPLE)
    (tmp_path / "later.csv").write_bytes(SPACESHIP_SAMPLE)  # the last-written rule: not here
    os.utime(elsewhere, (1, 1))
    code, out = _validate(tmp_path, contract, elsewhere)
    assert code == 0 and f"only files in {folder} are graded" in out and "later" not in out
    code, out = _validate(tmp_path, contract, folder / "submission.npy")
    assert code == 1 and "does not exist" in out


def _hps_folder(tmp_path: Path) -> tuple[dict, Path]:
    sample = b"row_id,target\n0-1,0.0\n0-2,0.0\n"
    contract = _contract(tmp_path, _competition(tmp_path, HPS_ID, sample))
    folder = tmp_path / "submission"
    for name in ("baseline_submission.csv", "submission.csv"):
        (folder / name).write_text("row_id,target\n0-1,0.1\n0-2,0.2\n")
    (folder / "baseline_model.pkl").write_bytes(b"opaque")
    return contract, folder


def test_hyperparameter_search_folders_need_their_models_and_code(tmp_path):
    contract, folder = _hps_folder(tmp_path)
    code, out = _validate(tmp_path, contract)
    assert code == 1 and "has no model.pkl" in out
    (folder / "model.pkl").write_bytes(b"opaque")  # never loaded by the check
    code, out = _validate(tmp_path, contract)
    assert code == 1 and "has no .py file" in out
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "search.py").write_text("import optuna\n")
    assert _validate(tmp_path, contract)[0] == 0
    (folder / "submission.csv").write_text("row_id,target\n0-1,0.1\n")
    code, out = _validate(tmp_path, contract, folder / "submission.csv")
    assert code == 1 and "1 rows" in out
    # another file: checked as a prediction file, as the grading server did
    (tmp_path / "predictions.csv").write_text("row_id,target\n0-1,0.1\n0-2,0.2\n")
    code, out = _validate(tmp_path, contract, tmp_path / "predictions.csv")
    assert code == 0 and "checked as a prediction file" in out
    code, out = _validate(tmp_path, contract, tmp_path / "contract.json")
    assert code == 1 and "not one of the files" in out


def test_feature_enhancement_folders_need_their_project_files(tmp_path):
    contract = _contract(tmp_path, _competition(tmp_path, FE_ID, b"label\n0\n0\n"))
    folder = tmp_path / "submission"
    (folder / "submission.csv").write_text("label\n1\n0\n")
    code, out = _validate(tmp_path, contract)
    assert code == 1 and "has no model.py, train.py" in out
    (tmp_path / "code").mkdir()
    for name in ("model.py", "train.py"):  # read from the code folder, as by make_submission.py
        (tmp_path / "code" / name).write_text("x = 1\n")
    code, out = _validate(tmp_path, contract)
    assert code == 0 and "conf/config.yaml does not exist" in out


# ---------------------------------------------------------------- npy, h5, pkl, py


def test_npy_submissions_must_have_the_shape_of_the_sample(tmp_path):
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((3, 4)))
    competition = _competition(tmp_path, "moment-anomaly-detection-score", buffer.getvalue())
    contract = _contract(tmp_path, competition)
    assert contract["file"]["shape"] == [3, 4]
    submission = tmp_path / "submission" / "submission.npy"
    for array, error in [
        (np.ones((3, 4), dtype=np.float32), None),
        (np.ones((4, 3)), "shape (4, 3)"),
        (np.full((3, 4), np.nan), "NaN"),
        (np.array([[None] * 4] * 3, dtype=object), "cannot read"),
    ]:
        np.save(submission, array, allow_pickle=True)
        code, out = _validate(tmp_path, contract)
        assert (code == 0) if error is None else (code == 1 and error in out), out


def _h5(path: Path, datasets: dict, link: str | None = None) -> None:
    with h5py.File(path, "w") as f:
        group = f.create_group("submission")
        for name, shape in datasets.items():
            group.create_dataset(name, data=np.zeros(shape))
        if link == "soft":
            group["c"] = h5py.SoftLink("/submission/a")
        if link == "external":
            group["c"] = h5py.ExternalLink("/private/answers.h5", "/labels")


def test_h5_submissions_must_have_the_datasets_of_the_sample(tmp_path):
    competition = _competition(tmp_path, "ucr-anomaly-detection")
    competition.sample_submission.parent.mkdir(parents=True)
    _h5(competition.sample_submission, {"a": (5,), "b": (7,)})
    contract = _contract(tmp_path, competition)
    assert contract["file"]["datasets"] == {
        "submission/a": {"shape": [5], "fields": None},
        "submission/b": {"shape": [7], "fields": None},
    }
    submission = tmp_path / "submission" / "submission.h5"
    for datasets, link, error in [
        ({"a": (5,), "b": (7,)}, None, None),
        ({"a": (5,)}, None, "1 missing"),
        ({"a": (5,), "b": (8,)}, None, "shape [8]"),
        ({"a": (5,), "b": (7,)}, "soft", "SoftLink"),
        ({"a": (5,), "b": (7,)}, "external", "ExternalLink"),
    ]:
        _h5(submission, datasets, link)
        code, out = _validate(tmp_path, contract)
        assert (code == 0) if error is None else (code == 1 and error in out), out


def test_h5_compound_datasets_are_checked_by_field(tmp_path):
    competition = _competition(tmp_path, "gift-eval-short-horizon")
    competition.sample_submission.parent.mkdir(parents=True)
    dtype = np.dtype([("m4_daily", "<f4", (2, 3)), ("solar/W", "<f4", (1, 3))])
    with h5py.File(competition.sample_submission, "w") as f:
        f.create_dataset("submission", data=np.zeros(1, dtype=dtype))
    contract = _contract(tmp_path, competition)
    fields = {"m4_daily": [2, 3], "solar/W": [1, 3]}
    assert contract["file"]["datasets"] == {"submission": {"shape": [1], "fields": fields}}
    submission = tmp_path / "submission" / "submission.h5"
    data = np.zeros(1, dtype=dtype)
    data["solar/W"][0, 0, 1] = np.inf
    with h5py.File(submission, "w") as f:
        f.create_dataset("submission", data=data)
    code, out = _validate(tmp_path, contract)
    assert code == 1 and "'solar/W'" in out and "infinite" in out


class _Payload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


def test_pickled_submissions_must_have_the_structure_of_the_sample(tmp_path):
    sample = {"TaskA": [{"seed": 1, "score": 0}, {"seed": 2, "score": 0}]}
    competition = _competition(tmp_path, "context-is-key-nn5-chronos", pickle.dumps(sample))
    contract = _contract(tmp_path, competition)
    submission = tmp_path / "submission" / "submission.pkl"
    marker = tmp_path / "pwned"
    for obj, error in [
        ({"TaskA": [{"seed": 1, "score": 0.3}, {"seed": 2, "score": np.float64(0.1)}]}, None),
        ({"TaskB": [{"seed": 1, "score": 0.3}, {"seed": 2, "score": 0.1}]}, "1 missing"),
        ({"TaskA": [{"seed": 1, "score": 0.3}]}, "1 items"),
        ({"TaskA": [{"seed": 1, "score": "x"}, {"seed": 2, "score": 0.1}]}, "a str"),
        ({"TaskA": np.zeros(2)}, "multiarray._reconstruct"),
        (_Payload(marker), "pathlib"),
    ]:
        submission.write_bytes(pickle.dumps(obj))
        code, out = _validate(tmp_path, contract)
        assert (code == 0) if error is None else (code == 1 and error in out), out
    assert not marker.exists()


def test_the_check_allows_the_pickle_globals_the_grader_allows():
    from timeseriesgym.utils import _PLAIN_PICKLE_GLOBALS

    assert vs.PLAIN_PICKLE_GLOBALS == _PLAIN_PICKLE_GLOBALS


def test_code_submissions_are_parsed_and_their_functions_listed(tmp_path):
    competition = _competition(tmp_path, "stomp-R-to-python", b"def stomp():\n    pass\n")
    contract = _contract(tmp_path, competition)
    assert contract["file"]["functions"]["stomp"] == {
        "args": ["ref_data", "query_data", "window_size", "exclusion_zone", "verbose"],
        "exact": True,
    }
    submission = tmp_path / "submission" / "submission.py"
    marker = tmp_path / "pwned"
    submission.write_text(f"import pathlib\npathlib.Path({str(marker)!r}).touch()\n")
    code, out = _validate(tmp_path, contract)
    assert code == 0 and "function stomp is not defined" in out and not marker.exists()
    submission.write_text("def stomp(ref_data, window_size):\n    pass\n")
    code, out = _validate(tmp_path, contract)
    assert code == 0 and "stomp must take exactly" in out
    args = "ref_data, query_data, window_size, exclusion_zone, verbose"
    submission.write_text(f"def stomp({args}):\n    pass\n")
    code, out = _validate(tmp_path, contract)
    assert code == 0 and "Note" not in out
    submission.write_text("def stomp(:\n")
    code, out = _validate(tmp_path, contract)
    assert code == 1 and "not valid Python" in out


def test_challenges_without_a_sample_file_are_not_checked(tmp_path):
    competition = _competition(tmp_path, "csdi-implement-model-pm25")
    contract = _contract(tmp_path, competition)  # its sample submission is a directory
    assert contract["kind"] == "none"
    code, out = _validate(tmp_path, contract)
    assert code == 0 and "not checked" in out


def test_the_format_check_runs_without_timeseriesgym(tmp_path):
    # it runs in the agent's image, which has no TimeSeriesGym code: `python -I`, another cwd
    source = (REPO / "environment" / "validate_submission.py").read_text()
    assert not re.search(r"^\s*(from|import) (timeseriesgym|agents|environment)", source, re.M)
    contract = _contract(tmp_path, _competition(tmp_path, "spaceship-titanic", SPACESHIP_SAMPLE))
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    (tmp_path / "submission" / "submission.csv").write_bytes(SPACESHIP_SAMPLE)
    out = subprocess.run(
        [sys.executable, "-I", str(REPO / "environment" / "validate_submission.py")],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"], "TIMESERIESGYM_CONTRACT": str(tmp_path / "contract.json")},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0 and out.stdout.startswith(vs.VALID), out.stderr
