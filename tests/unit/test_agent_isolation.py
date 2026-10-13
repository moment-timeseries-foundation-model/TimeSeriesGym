"""The agent's container has only public data and a format check; runs are graded afterwards in
a separate container with the private data (agents/run.py, agents/validation.py,
agents/grading.py, environment/validate_submission.py)."""

import asyncio
import importlib
import io
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import docker
import h5py
import numpy as np
import pytest
import requests

from agents.grading import DATA_DIR, grade_run
from agents.validation import OutdatedImageError, install_contract, submission_contract
from environment import validate_submission as vs
from timeseriesgym.registry import Registry

REPO = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("test")
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
    # a compound sample, as GIFT-Eval's was before it took a predictor
    competition = _competition(tmp_path, "ucr-anomaly-detection")
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


def _predictor_competition(tmp_path: Path):
    """gift-eval-short-horizon's public data as far as the format check reads them: the
    sample predictor and the grader's format check in predictor_tools/."""
    competition = _competition(tmp_path, "gift-eval-short-horizon", b"def predict(windows):\n")
    tools = competition.public_dir / "predictor_tools"
    tools.mkdir()
    (tools / "contract.py").write_bytes((REPO / "timeseriesgym/predictor/contract.py").read_bytes())
    return competition


def test_predictor_submissions_are_checked_by_the_graders_format_check(tmp_path):
    contract = _contract(tmp_path, _predictor_competition(tmp_path))
    # was kind "none": the sample, sample_predictor/predict.py, is not a file of predictions
    assert contract["kind"] == "predictor"
    assert contract["predictor"]["dir"] == "predictor"
    predictor = tmp_path / "submission" / "predictor"
    predictor.mkdir()
    marker = tmp_path / "ran"
    (predictor / "predict.py").write_text(
        f"open({str(marker)!r}, 'w').close()\n\ndef predict(windows):\n    return []\n"
    )
    (predictor / "weights.bin").write_bytes(b"\0" * 8)
    for path in (None, predictor, predictor / "predict.py"):  # the submission folder by default
        code, out = _validate(tmp_path, contract, path)
        assert code == 0 and out.startswith(vs.VALID) and "check_predictor.py" in out, out
    assert not marker.exists()  # parsed, never run

    (predictor / "link").symlink_to(tmp_path / "ran")
    code, out = _validate(tmp_path, contract)
    assert code == 1 and "symlink" in out
    (predictor / "link").unlink()
    (predictor / "predict.py").write_text("def forecast(windows):\n    return []\n")
    code, out = _validate(tmp_path, contract)
    assert code == 1 and "function `predict`" in out
    (tmp_path / "submission" / "submission.h5").write_bytes(b"")
    code, out = _validate(tmp_path, contract, tmp_path / "submission" / "submission.h5")
    assert code == 1 and "takes a predictor" in out


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


# ---------------------------------------------------------------- the agent's container


class _FakeExecAPI:
    def exec_create(self, container_id, cmd, user):
        return {"Id": "exec-1"}

    def exec_start(self, exec_id, stream):
        yield b"agent output\n"

    def exec_inspect(self, exec_id):
        return {"ExitCode": 0}


class _FakeAgentContainer:
    def __init__(self, put_archive=None):
        self.id, self.name = "c1", "competition-x"
        self.client = SimpleNamespace(api=_FakeExecAPI())
        self.events, self.archives = [], []
        self._put_archive = put_archive

    def put_archive(self, path, data):
        self.events.append("put_archive")
        if self._put_archive:
            return self._put_archive(path, data)
        self.archives.append((path, data))
        return True

    def start(self):
        self.events.append("start")

    def exec_run(self, cmd):
        self.events.append(("exec_run", cmd))
        return 0, b""

    def stop(self):
        self.events.append("stop")

    def remove(self):
        self.events.append("remove")


def _run_agent_phase(tmp_path, monkeypatch, container) -> tuple[dict, list]:
    agents_run = importlib.import_module("agents.run")
    created = {}

    def create(**kwargs):
        created.update(kwargs)
        return container

    monkeypatch.setattr(agents_run, "create_competition_container", create)
    monkeypatch.setattr(agents_run, "save_output", lambda c, run_dir, cfg: None)
    competition = _competition(tmp_path, "spaceship-titanic", SPACESHIP_SAMPLE)
    competition.private_dir.mkdir(parents=True)
    competition.answers.write_text("PassengerId,Transported\n0001_01,True\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent = SimpleNamespace(kwargs_type=None, kwargs={}, env_vars={"K": "v"}, privileged=False)
    exit_code = agents_run.run_in_container(
        client=None,
        competition=competition,
        agent=agent,
        image="img",
        container_config={},
        retain_container=False,
        run_dir=run_dir,
        logger=LOGGER,
    )
    assert exit_code == 0
    return created, competition


def test_the_agents_container_gets_only_the_public_data(tmp_path, monkeypatch):
    container = _FakeAgentContainer()
    created, competition = _run_agent_phase(tmp_path, monkeypatch, container)
    # the answers were mounted at /private/data/<id>/prepared/private, behind a mode-700 directory
    public = competition.public_dir.resolve().as_posix()
    assert created["volumes_config"] == {public: {"bind": "/home/data", "mode": "ro"}}
    passed = {k: v for k, v in created.items() if k != "competition"}
    assert str(competition.private_dir) not in json.dumps(passed, default=str)
    assert created["env_vars"] == {"COMPETITION_ID": "spaceship-titanic", "K": "v"}

    # the format check's contract, written into the created container before it starts
    assert container.events[:2] == ["put_archive", "start"]
    ((path, data),) = container.archives
    assert path == "/opt/timeseriesgym"
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        (member,) = tar.getmembers()
        contract = json.loads(tar.extractfile(member).read())
    assert member.name == "contract.json" and member.mode == 0o644
    assert contract["data_dir"] == "/home/data" and contract["submission_dir"] == "/home/submission"
    assert contract["file"]["id_columns"] == ["PassengerId"]
    saved = json.loads((tmp_path / "run" / "validation_contract.json").read_text())
    assert saved == contract
    # no grading server to wait for; the agent starts once the entrypoint has prepared /home
    (_, wait) = next(e for e in container.events if isinstance(e, tuple))
    assert "/run/timeseriesgym-ready" in wait[-1] and "5000" not in " ".join(wait)


def test_images_from_before_the_format_check_are_refused(tmp_path, monkeypatch):
    def missing(path, data):
        raise docker.errors.NotFound("Could not find the file /opt/timeseriesgym in container")

    container = _FakeAgentContainer(put_archive=missing)
    with pytest.raises(OutdatedImageError, match="Rebuild timeseriesgym-env"):
        _run_agent_phase(tmp_path, monkeypatch, container)
    assert container.events == ["put_archive", "stop", "remove"]  # the agent never started


def test_the_contract_is_installed_as_a_file(tmp_path):
    container = _FakeAgentContainer()
    install_contract(container, {"kind": "none", "note": "x"})
    ((_, data),) = container.archives
    out = tmp_path / "out"
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(out, filter="data")
    assert json.loads((out / "contract.json").read_text()) == {"kind": "none", "note": "x"}


# ---------------------------------------------------------------- the grading step


class _FakeGrader:
    """A grader container: on `wait`, `behaviour` writes a report as `timeseriesgym grade` does."""

    def __init__(self, behaviour):
        self.behaviour, self.events, self.name = behaviour, [], "grader-x"

    def start(self):
        self.events.append("start")

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        return {"StatusCode": self.behaviour(self.kwargs)}

    def logs(self):
        return b"graded\n"

    def kill(self):
        self.events.append("kill")

    def remove(self, force):
        self.events.append("remove")


def _grade(tmp_path, competition, behaviour, container_config=None, **kw):
    grader = _FakeGrader(behaviour)

    def create(**kwargs):
        grader.kwargs = kwargs
        return grader

    client = SimpleNamespace(containers=SimpleNamespace(create=create))
    run_dir = tmp_path / "runs" / "2026-01-01T00-00-00-GMT_run-group_agent:tag" / "run"
    (run_dir / "submission").mkdir(parents=True, exist_ok=True)
    result = grade_run(
        client=client,
        competition=competition,
        run_dir=run_dir,
        image="grader-image",
        container_config=container_config or {},
        logger=LOGGER,
        **kw,
    )
    return result, grader, run_dir


def _write_report(kwargs, valid=True) -> int:
    output_dir = Path(kwargs["command"][kwargs["command"].index("--output-dir") + 1])
    report = {"competition_reports": [{"score": 0.5, "valid_submission": valid}]}
    (output_dir / "20260101_grading_report.json").write_text(json.dumps(report))
    return 0


def test_runs_are_graded_in_a_separate_container_with_the_private_data(tmp_path):
    competition = _competition(tmp_path, "spaceship-titanic", SPACESHIP_SAMPLE)
    config = {"runtime": "sysbox-runc", "gpus": "0", "mem_limit": "16g", "nano_cpus": 4e9}
    result, grader, run_dir = _grade(tmp_path, competition, _write_report, config, timeout=60)
    run_dir = run_dir.resolve()
    kwargs = grader.kwargs
    assert result["valid_submission"] and result["score"] == 0.5
    assert result["report_path"] == str(run_dir / "grading" / "20260101_grading_report.json")
    prepared = f"{DATA_DIR}/spaceship-titanic/prepared"
    mounts = {(m["Source"], m["Target"], m["ReadOnly"]) for m in kwargs["mounts"]}
    assert mounts == {
        (str(competition.private_dir.resolve()), f"{prepared}/private", True),
        (str(competition.public_dir.resolve()), f"{prepared}/public", True),
        (str(run_dir), str(run_dir), True),
        (str(run_dir / "grading"), str(run_dir / "grading"), False),
    }
    assert {m["Type"] for m in kwargs["mounts"]} == {"bind"} and "volumes" not in kwargs
    assert kwargs["network_mode"] == "none" and kwargs["image"] == "grader-image"
    assert kwargs["entrypoint"] == ["timeseriesgym"]
    jsonl = run_dir / "grading" / "submission.jsonl"
    assert kwargs["command"] == [
        "grade",
        "--submission",
        str(jsonl),
        "--output-dir",
        str(run_dir / "grading"),
        "--data-dir",
        DATA_DIR,
    ]
    # the line make_submission.py writes for the run (no submission: graded as missing)
    line = {"competition_id": "spaceship-titanic", "submission_path": str(run_dir / "submission")}
    line["submission_path"] += "/submission.csv"
    assert json.loads(jsonl.read_text()) == line
    assert kwargs["user"] == f"{os.getuid()}:{os.getgid()}"
    # resource limits, but not the agent's runtime, GPUs or privileges
    assert kwargs["mem_limit"] == "16g" and kwargs["nano_cpus"] == 4_000_000_000
    assert not {"runtime", "device_requests", "privileged"} & set(kwargs)
    assert grader.events == ["start", ("wait", 60), "remove"]
    assert (run_dir / "grading" / "grader.log").read_text() == "graded\n"


def test_hyperparameter_search_runs_are_graded_with_the_line_of_make_submission(tmp_path):
    competition = _competition(tmp_path, HPS_ID, b"row_id,target\n0-1,0.0\n")
    result, grader, run_dir = _grade(tmp_path, competition, _write_report)
    line = json.loads((run_dir / "grading" / "submission.jsonl").read_text())
    assert line["solution_file_path"] == str(run_dir.resolve() / "code")
    assert line["submission_path"] == str(run_dir.resolve() / "submission")


@pytest.mark.parametrize("failure", ["exit", "timeout", "no report"])
def test_grading_failures_are_reported_not_raised(tmp_path, failure):
    competition = _competition(tmp_path, "spaceship-titanic", SPACESHIP_SAMPLE)

    def behaviour(kwargs):
        if failure == "timeout":
            raise requests.exceptions.ReadTimeout("read timed out")
        return 1 if failure == "exit" else 0

    result, grader, _ = _grade(tmp_path, competition, behaviour)
    expected = {"exit": "exited with code 1", "timeout": "took more", "no report": "no report"}
    assert expected[failure] in result["error"]
    assert grader.events[-1] == "remove"
    assert ("kill" in grader.events) is (failure == "timeout")


# ---------------------------------------------------------------- run_agent.py


def _run_worker(monkeypatch, tmp_path, run, grade, skip_grading=False) -> dict:
    run_agent = importlib.import_module("run_agent")
    args = SimpleNamespace(
        retain=False, skip_grading=skip_grading, grader_image="g", grading_timeout=7
    )
    monkeypatch.setattr(run_agent, "args", args, raising=False)
    monkeypatch.setattr(run_agent, "run_in_container", run)
    monkeypatch.setattr(run_agent, "grade_run", grade)
    task = run_agent.Task(
        run_id="run-1",
        seed=0,
        image="img",
        path_to_run_group=tmp_path,
        path_to_run=tmp_path,
        agent=SimpleNamespace(id="agent", name="img"),
        competition=SimpleNamespace(id="x"),
        container_config={"mem_limit": "1g"},
    )
    outputs = {}

    async def run_one():
        queue = asyncio.Queue()
        queue.put_nowait(task)
        worker = asyncio.create_task(run_agent.worker(0, queue, "client", outputs))
        await queue.join()
        worker.cancel()

    asyncio.run(run_one())
    return outputs["run-1"]


def test_each_run_is_graded_after_its_agent_phase(tmp_path, monkeypatch):
    events = []

    def run(**kwargs):
        events.append("agent")
        return 0

    def grade(**kwargs):
        events.append(("grade", kwargs["image"], kwargs["timeout"], kwargs["container_config"]))
        return {"score": 0.5, "valid_submission": True}

    output = _run_worker(monkeypatch, tmp_path, run, grade)
    assert events == ["agent", ("grade", "g", 7, {"mem_limit": "1g"})]
    assert output["success"] and output["grading"] == {"score": 0.5, "valid_submission": True}

    events.clear()
    assert "grading" not in _run_worker(monkeypatch, tmp_path, run, grade, skip_grading=True)
    assert events == ["agent"]


def test_failed_agent_phases_are_graded_too(tmp_path, monkeypatch):
    def run(**kwargs):
        raise RuntimeError("container failed")

    output = _run_worker(monkeypatch, tmp_path, run, lambda **kw: {"error": "RuntimeError: x"})
    # every run counts, as in make_submission.py: a run without a submission is graded invalid
    assert output["success"] is False and output["grading"] == {"error": "RuntimeError: x"}


def test_run_agent_needs_the_grader_image(tmp_path, monkeypatch):
    run_agent = importlib.import_module("run_agent")

    def missing(name):
        raise docker.errors.ImageNotFound(name)

    client = SimpleNamespace(images=SimpleNamespace(get=missing))
    monkeypatch.setattr(run_agent.docker, "from_env", lambda: client)
    monkeypatch.setattr(run_agent, "registry", run_agent.registry)  # main() replaces it
    args = SimpleNamespace(
        data_dir=str(tmp_path), agent_id="dummy", skip_grading=False, grader_image="missing"
    )
    with pytest.raises(ValueError, match="--skip-grading"):
        asyncio.run(run_agent.main(args))


# ---------------------------------------------------------------- images


def _stages(dockerfile: str) -> dict[str, str]:
    """The Dockerfile's stages by name (the unnamed last one, the default target, is '')."""
    parts = re.split(r"^FROM ", dockerfile, flags=re.M)[1:]
    stages = {}
    for part in parts:
        first = part.splitlines()[0]
        name = first.split(" AS ")[1].strip() if " AS " in first else ""
        stages[name] = part
    return stages


def test_the_agents_image_has_no_graders_or_private_data():
    stages = _stages((REPO / "environment" / "Dockerfile").read_text())
    agent, grader = stages[""], stages["grader"]
    assert list(stages)[-1] == ""  # `docker build -f environment/Dockerfile .` builds the agent's
    for leak in ("COPY . ", "pip install -e", "/private", "grading_server", "flask"):
        assert leak not in agent
    tool = vs.CONTRACT_PATH.parent / "validate_submission.py"
    assert f"COPY environment/validate_submission.py {tool}" in agent
    assert "COPY . /timeseriesgym" in grader and 'ENTRYPOINT ["timeseriesgym"]' in grader
    entrypoint = (REPO / "environment" / "entrypoint.sh").read_text()
    assert "grading_server" not in entrypoint and "/private" not in entrypoint
    agents_run = importlib.import_module("agents.run")
    assert f"touch {agents_run.READY_FILE}" in entrypoint
    wrapper = (REPO / "environment" / "validate_submission.sh").read_text()
    assert f"exec /opt/conda/bin/python -I {tool} " in wrapper and "5000" not in wrapper
