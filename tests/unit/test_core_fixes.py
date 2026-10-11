"""Regression tests for fixes in the core package, the container environment and the runners."""

import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import threading
import time
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from timeseriesgym.grade import grade_code, grade_jsonl, validate_submission
from timeseriesgym.registry import Registry
from timeseriesgym.utils import (
    InvalidSubmissionError,
    TorchModel,
    have_different_hyperparameters,
    have_hyperparameter_searching_attempt,
    load_models,
    load_scripts,
    load_submissions,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- loaders


@pytest.mark.parametrize("load", [load_submissions, load_scripts, load_models])
def test_loaders_raise_for_missing_files(tmp_path, load):
    # they used to return the exception, which graders then used as data
    with pytest.raises(InvalidSubmissionError, match="does not exist"):
        load(tmp_path / "missing.csv")


class _Payload:
    """Pickles to a call of `Path.touch`: unpickling it creates the marker file."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


def _cik_like_result():
    result = defaultdict(list)
    for task in ("TaskA", "TaskB"):
        for seed in range(1, 4):
            result[task].append({"seed": seed, "score": np.float64(seed / 7)})
    return result


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_pickled_submissions_of_plain_data_load(tmp_path, protocol):
    path = tmp_path / "submission.pkl"
    result = _cik_like_result()
    result["other"] = [np.arange(3.0), np.ones((2, 2), dtype=np.int32), {1, 2}, (1.5, None)]
    path.write_bytes(pickle.dumps(result, protocol=protocol))
    loaded = load_submissions(path)
    assert isinstance(loaded, defaultdict) and loaded["TaskA"] == result["TaskA"]
    assert np.array_equal(loaded["other"][1], result["other"][1])
    assert loaded["other"][2:] == [{1, 2}, (1.5, None)]


def test_pickled_submissions_cannot_run_code(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "submission.pkl"
    path.write_bytes(pickle.dumps({"TaskA": [_Payload(marker)]}))
    with pytest.raises(InvalidSubmissionError, match="references"):
        load_submissions(path)
    assert not marker.exists()


# ---------------------------------------------------------------- validation (grading server)


def _prepared(data_dir: Path, competition_id: str, answers: bytes):
    """A competition whose private answers are `answers`, as `is_dataset_prepared` expects."""
    competition = Registry(data_dir).get_competition(competition_id)
    competition.answers.parent.mkdir(parents=True, exist_ok=True)
    competition.answers.write_bytes(answers)
    return competition


def test_validation_only_parses_code_submissions(tmp_path):
    competition = _prepared(tmp_path / "data", "stomp-R-to-python", b"a,b\n1,2\n")
    marker = tmp_path / "pwned"
    submission = tmp_path / "submission.py"
    submission.write_text(f"import pathlib\npathlib.Path({str(marker)!r}).touch()\n")
    valid, message = validate_submission(submission, competition)
    assert valid and "Submission is valid." in message
    assert not marker.exists()  # the grading server runs as root: importing it ran this

    submission.write_text("def stomp(:\n")
    valid, message = validate_submission(submission, competition)
    assert not valid and "not valid Python" in message


def test_validation_does_not_unpickle_code(tmp_path):
    answers = pickle.dumps(_cik_like_result())
    competition = _prepared(tmp_path / "data", "context-is-key-moirai", answers)
    submission = tmp_path / "submission.pkl"
    submission.write_bytes(answers)
    assert validate_submission(submission, competition) == (True, "Submission is valid.")

    marker = tmp_path / "pwned"
    submission.write_bytes(pickle.dumps({"TaskA": [_Payload(marker)]}))
    valid, message = validate_submission(submission, competition)
    assert not valid and "references" in message
    assert not marker.exists()


SPACESHIP_ANSWERS = b"PassengerId,Transported\n0001_01,True\n0002_01,False\n0003_01,True\n"


def test_validation_reports_the_grader_result(tmp_path, monkeypatch):
    competition = _prepared(tmp_path / "data", "spaceship-titanic", SPACESHIP_ANSWERS)
    submission = tmp_path / "submission.csv"
    submission.write_bytes(SPACESHIP_ANSWERS)
    # was "unexpected keyword argument 'submission_file'" for every file
    assert validate_submission(submission, competition) == (True, "Submission is valid.")

    submission.write_text("PassengerId,Prediction\n0001_01,True\n0002_01,False\n0003_01,True\n")
    valid, message = validate_submission(submission, competition)
    assert not valid and "Transported" in message

    for score in (None, float("nan")):  # was ignored: everything that loaded was "valid"
        monkeypatch.setattr(competition.grader, "grade_fn", lambda submission, answers, s=score: s)
        valid, message = validate_submission(submission, competition)
        assert not valid and "could not score" in message


HPS_ID = "optiver-realized-volatility-prediction-hyperparameter-search"
OPTIVER_ANSWERS = b"row_id,target\n0-1,0.002\n0-2,0.003\n"


def test_validation_of_hyperparameter_search_folders(tmp_path):
    competition = _prepared(tmp_path / "data", HPS_ID, OPTIVER_ANSWERS)
    folder = tmp_path / "submission"
    folder.mkdir()
    for name in ("baseline_submission.csv", "submission.csv"):
        (folder / name).write_bytes(OPTIVER_ANSWERS)
    marker = tmp_path / "pwned"
    (folder / "baseline_model.pkl").write_bytes(pickle.dumps(_Payload(marker)))
    valid, message = validate_submission(folder, competition)
    assert not valid and "model.pkl" in message  # was "does not exist" for every folder

    (folder / "model.pkl").write_bytes(pickle.dumps(_Payload(marker)))
    valid, message = validate_submission(folder, competition)
    assert valid and "Submission is valid." in message
    assert not marker.exists()  # models are not unpickled by the root server

    (folder / "submission.csv").write_text("row_id,target\n0-1,0.002\n")
    valid, message = validate_submission(folder, competition)
    assert not valid and message.startswith("submission.csv: Submission invalid!")

    # the grading server receives single files: validated as one of the prediction files
    assert validate_submission(folder / "baseline_submission.csv", competition)[0]
    assert not validate_submission(folder / "submission.csv", competition)[0]


def test_validation_of_feature_enhancement_prediction_files(tmp_path):
    answers = b"ecg_id,label\n1,NORM\n2,MI\n"
    competition = _prepared(
        tmp_path / "data", "ptb-xl-classification-challenge-feature-enhancement", answers
    )
    submission = tmp_path / "submission.csv"
    submission.write_bytes(answers)
    assert validate_submission(submission, competition) == (True, "Submission is valid.")
    submission.write_text("ecg_id,prediction\n1,NORM\n2,MI\n")
    assert not validate_submission(submission, competition)[0]


def test_grading_server_saves_each_upload_in_a_private_directory(tmp_path, monkeypatch):
    # flask is not installed in the test image: stand in for the three names the server uses
    flask = types.ModuleType("flask")
    flask.Flask = lambda name: SimpleNamespace(route=lambda *a, **k: (lambda fn: fn))
    flask.jsonify = lambda payload: payload
    flask.request = SimpleNamespace(files={})
    monkeypatch.setitem(sys.modules, "flask", flask)
    monkeypatch.delitem(sys.modules, "environment.grading_server", raising=False)
    server = importlib.import_module("environment.grading_server")

    seen = []

    def run_validation(path: Path) -> str:
        seen.append((path, path.read_bytes(), oct(path.parent.stat().st_mode & 0o777)))
        return "checked"

    monkeypatch.setattr(server, "run_validation", run_validation)

    def save(path):
        Path(path).write_bytes(b"x,y\n")

    upload = SimpleNamespace(filename="my_submission.CSV", save=save)
    flask.request.files = {"file": upload}
    assert server.validate() == {"result": "checked"}
    assert server.validate() == {"result": "checked"}

    (first, data, mode), (second, _, _) = seen
    assert data == b"x,y\n" and first.suffix == ".csv" and mode == "0o700"
    assert first.parent != second.parent  # was the fixed /tmp/submission_to_validate.csv
    assert not first.exists() and not second.exists()

    flask.request.files = {"file": SimpleNamespace(filename="run.sh", save=save)}
    assert server.validate()[1] == 400


# ---------------------------------------------------------------- agent image build context


def _dockerignored(path: str, patterns: list[str]) -> bool:
    """Docker's .dockerignore rule: the last pattern matching the path or one of its parent
    directories decides; `!` re-includes; `*` stays within one path segment, `**` does not."""

    def regex(pattern: str) -> str:
        out = pattern.strip("/").replace(".", r"\.").replace("?", "[^/]")
        return out.replace("**/", "\0").replace("*", "[^/]*").replace("\0", "(.*/)?")

    parts = path.split("/")
    candidates = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    excluded = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        if any(re.fullmatch(regex(pattern.lstrip("!")), c) for c in candidates):
            excluded = not negated
    return excluded


def test_agent_image_context_excludes_answers_and_history():
    lines = (REPO / ".dockerignore").read_text().splitlines()
    patterns = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    leaked = [
        "timeseriesgym/data/additional_competition_data.zip",  # raw data incl. test answers
        "timeseriesgym/competitions/ptb-xl-classification-challenge-feature-enhancement/"
        "example/model.py",
        ".git/HEAD",
        "runs/run-group/run/submission/submission.csv",
        "data/spaceship-titanic/prepared/private/test.csv",
        ".env",
        "timeseriesgym/__pycache__/grade.cpython-311.pyc",
    ]
    needed = [
        "pyproject.toml",
        "README.md",  # pyproject's readme
        "environment/grading_server.py",
        "environment/entrypoint.sh",
        "timeseriesgym/grade.py",
        "timeseriesgym/competitions/spaceship-titanic/grade.py",
        "timeseriesgym/competitions/spaceship-titanic/description.md",
    ]
    assert [p for p in leaked if not _dockerignored(p, patterns)] == []
    assert [p for p in needed if _dockerignored(p, patterns)] == []


# ---------------------------------------------------------------- agents


def test_openhands_setup_script_runs(tmp_path):
    # start.sh runs it as `python setup.py`: a relative import failed before argparse ran
    env = {**os.environ, "CODE_DIR": str(tmp_path), "AGENT_DIR": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, "setup.py", "--help"],
        cwd=REPO / "agents" / "openhands",
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "--max_steps" in out.stdout


def test_openhands_root_build_script_is_not_agent_writable():
    openhands = REPO / "agents" / "openhands"
    dockerfile = (openhands / "Dockerfile").read_text()
    allowed = re.findall(r"NOPASSWD:\s*(\S+?)\"", dockerfile)
    assert allowed == ["/agent_build.sh"]
    # environment/entrypoint.sh makes everything under /home writable for nonroot
    assert "find /home " in (REPO / "environment" / "entrypoint.sh").read_text()
    assert not allowed[0].startswith(("/home/", "${AGENT_DIR}", "$AGENT_DIR"))
    assert "COPY build.sh /agent_build.sh" in dockerfile
    assert "sudo /agent_build.sh" in (openhands / "start.sh").read_text()
    # it builds from /home/agent, which nonroot can change: only the first call may build
    build = (openhands / "build.sh").read_text()
    assert build.index("mkdir /root/.agent_build_started") < build.index("make build")


# ---------------------------------------------------------------- registry and environment


def test_competition_ids_are_the_competition_directories():
    registry = Registry()
    ids = registry.list_competition_ids()
    expected = sorted(p.parent.name for p in registry.get_competitions_dir().glob("*/config.yaml"))
    assert ids == expected and "conf" not in ids  # was listed from the FE example's Hydra config
    for competition_id in ids:  # `prepare --all` and `cleanup --all` load every id
        assert registry.get_competition(competition_id).id == competition_id


def test_agent_environment_has_the_packages_of_shipped_helpers():
    requirements = (REPO / "environment" / "requirements.txt").read_text()
    names = {re.split(r"[=<>~ ]", line)[0].lower() for line in requirements.splitlines() if line}
    helper = REPO / "timeseriesgym/competitions/gift-eval-short-horizon/dataset.py"
    assert "from dotenv import" in helper.read_text()  # copied to the agent's /home/data
    assert "python-dotenv" in names


# ---------------------------------------------------------------- grading reports


def _jsonl(path: Path, entries: list[dict]) -> Path:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def test_unreadable_submissions_do_not_abort_a_grading_batch(tmp_path):
    data = tmp_path / "data"
    competition = _prepared(data, "spaceship-titanic", SPACESHIP_ANSWERS)
    good = tmp_path / "good.csv"
    good.write_bytes(SPACESHIP_ANSWERS)
    ragged = tmp_path / "ragged.csv"  # pandas ParserError
    ragged.write_text('PassengerId,Transported\n0001_01,True\n0002_01,"False,1\n0003_01,True,3,4\n')
    wrong_type = tmp_path / "submission.txt"
    wrong_type.write_bytes(SPACESHIP_ANSWERS)
    paths = [good, ragged, wrong_type, tmp_path / "missing.csv"]
    entries = [{"competition_id": competition.id, "submission_path": str(p)} for p in paths]

    out = tmp_path / "reports"
    out.mkdir()
    grade_jsonl(_jsonl(tmp_path / "s.jsonl", entries), out, Registry(data))
    (report_file,) = out.iterdir()
    reports = json.loads(report_file.read_text())["competition_reports"]
    assert [r["valid_submission"] for r in reports] == [True, False, False, False]
    assert [r["submission_exists"] for r in reports] == [True, True, True, False]
    assert float(reports[0]["score"]) == 1.0 and reports[1]["score"] is None


def test_rejected_code_submissions_get_an_invalid_report(tmp_path):
    competition = _prepared(tmp_path / "data", "stomp-R-to-python", b"a,b\n1,2\n")
    submission = tmp_path / "submission.py"
    submission.write_text("def stomp(:\n")  # the grader cannot import it and returns None
    report = grade_code(submission, competition)  # was TypeError: None["defined_classes"]
    assert not report.valid_submission and report.submission_exists
    assert report.test_metric is None and report.defined_functions is None

    other = tmp_path / "submission.txt"
    other.write_text("def stomp(): pass\n")
    assert not grade_code(other, competition).valid_submission  # raised before


def test_grading_report_directory_is_created_with_its_parents(tmp_path):
    data = tmp_path / "data"
    competition = _prepared(data, "spaceship-titanic", SPACESHIP_ANSWERS)
    entry = {"competition_id": competition.id, "submission_path": str(tmp_path / "none.csv")}
    out = tmp_path / "new" / "results"  # failed after grading: `new` did not exist
    grade_jsonl(_jsonl(tmp_path / "s.jsonl", [entry]), out, Registry(data))
    assert len(list(out.glob("*_grading_report.json"))) == 1


def test_dict_scores_are_reported_as_json_objects():
    from datetime import datetime

    from timeseriesgym.grade_helpers import CompetitionReport
    from timeseriesgym.utils import get_null_rank_info

    rank = {k: v for k, v in get_null_rank_info().items() if not k.endswith("_medal")}
    medals = dict.fromkeys(["any_medal", "gold_medal", "silver_medal", "bronze_medal"], False)
    fields = {**rank, **medals, "above_median": False, "is_lower_better": False}
    fields.update(submission_exists=True, valid_submission=True, created_at=datetime.now())
    score = {"accuracy": np.float64(0.5), "ETTh1": np.float32(0.25), "overall_score": 0.75}
    report = CompetitionReport(competition_id="x", score=score, submission_path="s", **fields)
    as_dict = json.loads(json.dumps(report.to_dict()))
    assert as_dict["score"] == {"accuracy": 0.5, "ETTh1": 0.25, "overall_score": 0.75}  # was a repr
    assert CompetitionReport.from_dict(as_dict).score == as_dict["score"]  # was float(dict)

    plain = CompetitionReport(competition_id="x", score=0.5, submission_path="s", **fields)
    assert plain.to_dict()["score"] == "0.5"  # unchanged for numeric scores
    assert CompetitionReport.from_dict(plain.to_dict()).score == 0.5


def test_make_submission_points_folder_challenges_at_their_folders(tmp_path):
    runs = {}
    for competition_id in (
        "spaceship-titanic",
        "ptb-xl-classification-challenge-feature-enhancement",
        HPS_ID,
    ):
        submission_dir = tmp_path / competition_id / "submission"
        submission_dir.mkdir(parents=True)
        (submission_dir / "submission.csv").write_text("id,label\n")
        run = {"competition_id": competition_id, "run_dir": str(submission_dir.parent)}
        runs[competition_id] = {**run, "success": True}
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"runs": runs}))
    output = tmp_path / "submission.jsonl"
    args = ["--metadata", metadata, "--output", output, "--data-dir", tmp_path / "data"]
    subprocess.run(
        [sys.executable, "experiments/make_submission.py", *map(str, args)],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        check=True,
        capture_output=True,
    )
    lines = {line["competition_id"]: line for line in map(json.loads, output.open())}
    root = {cid: tmp_path / cid / "submission" for cid in runs}
    assert lines["spaceship-titanic"]["submission_path"] == str(
        root["spaceship-titanic"] / "submission.csv"
    )
    fe = lines["ptb-xl-classification-challenge-feature-enhancement"]  # was .../submission.csv
    assert fe["submission_path"] == str(root[fe["competition_id"]])
    assert lines[HPS_ID] == {  # the search script was never passed, so never checked
        "competition_id": HPS_ID,
        "submission_path": str(root[HPS_ID]),
        "solution_file_path": str(root[HPS_ID] / "solution.py"),
    }


# ---------------------------------------------------------------- hyperparameter search


def _scripted(tmp_path: Path, name: str, **kwargs) -> TorchModel:
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self, stride=1, padding=0, act=None):
            super().__init__()
            self.conv = nn.Conv1d(12, 8, 3, stride=stride, padding=padding)
            self.act = act if act is not None else nn.ReLU()
            self.fc = nn.Linear(8, 5)

        def forward(self, x):
            return self.fc(self.act(self.conv(x)).mean(-1))

    path = tmp_path / f"{name}.pt"
    torch.jit.script(Net(**kwargs)).save(str(path))  # as the PTB-XL description asks
    return TorchModel.load(str(path))


def test_torchscript_models_compare_their_layer_hyperparameters(tmp_path):
    from torch import nn

    baseline = _scripted(tmp_path, "baseline")
    assert not have_different_hyperparameters(baseline, _scripted(tmp_path, "same"))
    # same parameter shapes; these used to count as "same hyperparameters" (score -inf)
    assert have_different_hyperparameters(baseline, _scripted(tmp_path, "stride", stride=2))
    assert have_different_hyperparameters(baseline, _scripted(tmp_path, "pad", padding=1))
    assert have_different_hyperparameters(baseline, _scripted(tmp_path, "gelu", act=nn.GELU()))


@pytest.mark.parametrize(
    "code,expected",
    [
        # common in any training script; each counted as a hyperparameter search
        ("from sklearn.model_selection import train_test_split\ntrain_test_split(x, y)\n", False),
        ("import sklearn.model_selection as ms\nms.KFold(5)\n", False),
        ("import wandb\nwandb.init(project='p')\n", False),
        ("import mlflow\nmlflow.log_metric('acc', 1.0)\n", False),
        ("import ray.tune\nimport ray\nray.init()\n", False),
        # searches
        (
            "from sklearn.model_selection import GridSearchCV\nGridSearchCV(m, grid).fit(x, y)\n",
            True,
        ),
        ("import sklearn.model_selection as ms\nms.RandomizedSearchCV(m, d)\n", True),
        ("from sklearn import model_selection\nmodel_selection.GridSearchCV(m, g)\n", True),
        ("import wandb\nwandb.sweep(config)\n", True),
        ("import ray\nray.tune.run(train)\n", True),
        ("from ray import tune\ntune.Tuner(train)\n", True),
        ("import optuna\nstudy = optuna.create_study()\n", True),
        ("from optuna.samplers import TPESampler\nTPESampler()\n", True),
        ("import optuna\n", False),
    ],
)
def test_default_hyperparameter_search_apis(code, expected):
    assert have_hyperparameter_searching_attempt(code) is expected


# ---------------------------------------------------------------- LLM judges


def test_judge_json_output_survives_parallel_evaluations(tmp_path, monkeypatch, capsys):
    judge = importlib.import_module("timeseriesgym.judge")
    a_running, b_running = threading.Event(), threading.Event()

    class FakeGEval:  # forces the interleaving: A starts, B starts, A ends, B ends
        def __init__(self, name, threshold, **kwargs):
            self.name, self.threshold, self.score, self.reason = name, threshold, None, "r"
            if name == "b":
                assert a_running.wait(5)

        def measure(self, test_case):
            print(f"deepeval noise from {self.name}")
            if self.name == "a":
                a_running.set()
                assert b_running.wait(5)
            else:
                b_running.set()
                time.sleep(0.2)
            self.score = 1.0

    monkeypatch.setattr(judge, "GEval", FakeGEval)
    monkeypatch.setattr(judge, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    artifacts = []
    for name in ("a", "b"):
        (tmp_path / name).write_text("code")
        criteria = [{"name": name, "description": "d"}]
        artifacts.append({"path": str(tmp_path / name), "type": "code", "criteria": criteria})
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"artifacts": artifacts}))  # JSON is YAML

    stdout = sys.stdout
    try:
        judge.llm_judge(str(config), "model", json_output=True)
        assert sys.stdout is stdout  # was left pointing at a worker's buffer
    finally:
        sys.stdout = stdout
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"] == {"total_artifacts": 2, "total_criteria": 2, "passed_criteria": 2}


def test_reasonableness_without_an_api_key_is_not_a_pass(tmp_path, monkeypatch, capsys):
    reasonableness = importlib.import_module("timeseriesgym.reasonableness")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(reasonableness, "load_dotenv", lambda: None)
    (tmp_path / "copy.py").write_text("import shutil\nshutil.copy('sample_submission.csv', 's')\n")
    result = reasonableness.evaluate_reasonableness(str(tmp_path))
    assert result.is_reasonable is None and result.score is None  # was True and 1.0
    assert "OPENAI_API_KEY" in result.reason

    from timeseriesgym import cli

    for extra in ([], ["--json"]):  # the CLI formatted the score with :.4f
        argv = ["timeseriesgym", "evaluate-reasonableness", str(tmp_path), *extra]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
    assert json.loads(capsys.readouterr().out)["is_reasonable"] is None


# ---------------------------------------------------------------- code grading helpers


def test_code_grading_data_of_unsupported_type_is_an_error(tmp_path):
    from timeseriesgym.grade_code_helper import load_dataset_for_function

    np.save(tmp_path / "x.npy", np.arange(3))
    (tmp_path / "y.csv").write_text("a\n1\n")
    config = {"required_args": {"x": {"file_name": "x.npy"}, "y": {"file_name": "y.csv"}}}
    with pytest.raises(ValueError, match="Unsupported file type"):  # y silently got x's array
        load_dataset_for_function(config, tmp_path)
