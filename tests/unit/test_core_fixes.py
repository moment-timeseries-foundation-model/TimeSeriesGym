"""Regression tests for fixes in the core package, the container environment and the runners."""

import functools
import importlib
import importlib.util
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
    result["other"] = [np.float32(2.5), np.int64(3), {1, 2}, frozenset("a"), (1.5, None, "s", b"b")]
    path.write_bytes(pickle.dumps(result, protocol=protocol))
    loaded = load_submissions(path)
    assert isinstance(loaded, defaultdict) and loaded == result
    assert type(loaded["TaskA"][0]["score"]) is np.float64


def test_pickled_submissions_cannot_hold_arrays(tmp_path):
    path = tmp_path / "submission.pkl"
    path.write_bytes(pickle.dumps({"TaskA": [np.zeros(3)]}))
    with pytest.raises(InvalidSubmissionError, match="references"):  # sized by the file
        load_submissions(path)


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

    code = tmp_path / "code"  # the solution code, which grading requires
    valid, message = validate_submission(folder, competition, solution_file_path=code)
    assert not valid and "solution code" in message
    code.mkdir()
    assert validate_submission(folder, competition, solution_file_path=code)[0]

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


def test_only_challenges_that_are_ranked_say_they_award_medals():
    from timeseriesgym.data import get_leaderboard

    registry = Registry()
    for competition_id in registry.list_competition_ids():
        competition = registry.get_competition(competition_id)
        graded_as_a_file = (
            competition.hyperparameter_search_config is None
            and competition.coding_config is None
            and competition.feature_enhancement_config is None
        )
        ranked = graded_as_a_file and get_leaderboard(competition) is not None
        # Optiver HPS said true, but HPS reports are never ranked (grade.py)
        assert ranked or not competition.awards_medals, competition_id


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
    assert lines[HPS_ID] == {  # the solution code was never passed, so never checked
        "competition_id": HPS_ID,
        "submission_path": str(root[HPS_ID]),
        "solution_file_path": str(tmp_path / HPS_ID / "code"),
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
        # a mention, or a search object that is never run, is not a search
        ("import optuna\nprint(optuna.__version__)\n", False),
        ("import optuna\noptuna.logging.set_verbosity(optuna.logging.WARNING)\n", False),
        ("import optuna\nstudy = optuna.create_study()\n", False),
        ("from optuna.samplers import TPESampler\nTPESampler()\n", False),
        ("from sklearn.model_selection import GridSearchCV\nGridSearchCV(m, grid)\n", False),
        ("import wandb\nwandb.sweep(config)\n", False),
        # searches
        (
            "from sklearn.model_selection import GridSearchCV\nGridSearchCV(m, grid).fit(x, y)\n",
            True,
        ),
        ("import sklearn.model_selection as ms\nms.RandomizedSearchCV(m, d).fit(x, y)\n", True),
        ("from sklearn import model_selection as ms\ns = ms.GridSearchCV(m, g)\ns.fit(x)\n", True),
        ("import wandb\nsweep_id = wandb.sweep(config)\nwandb.agent(sweep_id, train)\n", True),
        ("import ray\nray.tune.run(train)\n", True),
        ("from ray import tune\ntune.Tuner(train).fit()\n", True),
        ("from hyperopt import fmin, tpe\nfmin(f, space, algo=tpe.suggest)\n", True),
        ("import optuna\nstudy = optuna.create_study()\nstudy.optimize(f, n_trials=9)\n", True),
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


# ---------------------------------------------------------------- runners


def test_run_agent_rejects_zero_workers(tmp_path):
    competitions = tmp_path / "split.txt"
    competitions.write_text("spaceship-titanic\n")
    args = ["--agent-id", "dummy", "--n-workers", "0", "--competition-set", str(competitions)]
    out = subprocess.run(
        [sys.executable, "run_agent.py", *args],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 2 and "--n-workers must be at least 1" in out.stderr


class _FakeExecAPI:
    def __init__(self, exit_code: int):
        self.exit_code, self.created = exit_code, []

    def exec_create(self, container_id, cmd, user):
        self.created.append((container_id, cmd, user))
        return {"Id": "exec-1"}

    def exec_start(self, exec_id, stream):
        assert stream
        yield b"agent output\n"

    def exec_inspect(self, exec_id):
        return {"ExitCode": self.exit_code}


@pytest.mark.parametrize("exit_code", [0, 3, 124])
def test_runs_return_the_agents_exit_code(tmp_path, monkeypatch, exit_code):
    import logging

    agents_run = importlib.import_module("agents.run")
    api = _FakeExecAPI(exit_code)
    container = SimpleNamespace(
        id="c1",
        name="competition-x",
        client=SimpleNamespace(api=api),
        start=lambda: None,
        exec_run=lambda cmd: (0, b""),  # the grading server's health check
        stop=lambda: None,
        remove=lambda: None,
    )
    saved = []
    monkeypatch.setattr(agents_run, "create_competition_container", lambda **kw: container)
    monkeypatch.setattr(agents_run, "save_output", lambda c, run_dir, cfg: saved.append(run_dir))
    competition = SimpleNamespace(id="x", public_dir=tmp_path, private_dir=tmp_path)
    agent = SimpleNamespace(kwargs_type=None, kwargs={}, env_vars={}, privileged=False)
    run = functools.partial(
        agents_run.run_in_container,
        client=None,
        competition=competition,
        agent=agent,
        image="img",
        container_config={},
        retain_container=False,
        run_dir=tmp_path,
        logger=logging.getLogger("test"),
    )
    # ignored at first, then raised after the outputs were saved, which made run_agent.py and
    # make_submission.py drop a submission saved before a timeout's 124
    assert run() == exit_code
    assert saved == [tmp_path]  # outputs are extracted whatever the exit code
    assert api.created[0][2] == "nonroot"


@pytest.mark.parametrize("exit_code", [0, 124])
def test_run_metadata_records_the_agents_exit_code(tmp_path, monkeypatch, exit_code):
    import asyncio

    run_agent = importlib.import_module("run_agent")
    monkeypatch.setattr(run_agent, "args", SimpleNamespace(retain=False), raising=False)
    monkeypatch.setattr(run_agent, "run_in_container", lambda **kwargs: exit_code)
    task = run_agent.Task(
        run_id="run-1",
        seed=0,
        image="img",
        path_to_run_group=tmp_path,
        path_to_run=tmp_path,
        agent=SimpleNamespace(id="agent", name="img"),
        competition=SimpleNamespace(id="x"),
        container_config={},
    )
    outputs = {}

    async def run_one():
        queue = asyncio.Queue()
        queue.put_nowait(task)
        worker = asyncio.create_task(run_agent.worker(0, queue, None, outputs))
        await queue.join()
        worker.cancel()

    asyncio.run(run_one())
    assert outputs["run-1"]["agent_exit_code"] == exit_code
    assert outputs["run-1"]["success"] is (exit_code == 0)


# ---------------------------------------------------------------- container output extraction


def test_sysbox_extraction_handles_large_stream_chunks(tmp_path):
    import io
    import tarfile

    from environment.utils import extract_from_container_sysbox

    buf = io.BytesIO()
    payload = bytes(range(256)) * 200  # 51 kB, larger than tarfile's 10 kB reads
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("submission/submission.csv")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    data = buf.getvalue()
    chunks = [b"", data[:7], data[7:]]  # an empty chunk, a small one, then one large chunk
    container = SimpleNamespace(exec_run=lambda **kw: SimpleNamespace(output=iter(chunks)))
    extract_from_container_sysbox(container, "/home/submission", tmp_path)
    # a chunk larger than the read buffer raised ValueError, which the caller swallowed
    assert (tmp_path / "submission" / "submission.csv").read_bytes() == payload


# ---------------------------------------------------------------- familiarity experiment


@pytest.fixture
def relevant_content(monkeypatch):
    """experiments/familiarity/get_relevant_content.py, with its scraping libraries stubbed."""
    page = SimpleNamespace(close=lambda: None)
    browser = SimpleNamespace(new_page=lambda: page, close=lambda: None)
    playwright = SimpleNamespace(chromium=SimpleNamespace(launch=lambda headless: browser))

    class SyncPlaywright:
        def __enter__(self):
            return playwright

        def __exit__(self, *exc):
            return False

    stubs = {
        "git": {"Repo": object},
        "newspaper": {"Article": object},
        "readability": {"Document": object},
        "playwright": {},
        "playwright.sync_api": {"sync_playwright": SyncPlaywright},
        "get_discussion_posts": {"extract_text_from_div": lambda page, url: f"posts of {url}"},
    }
    for name, attrs in stubs.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    path = REPO / "experiments" / "familiarity" / "get_relevant_content.py"
    spec = importlib.util.spec_from_file_location("get_relevant_content", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arxiv_sources_cannot_write_outside_their_directory(
    tmp_path, monkeypatch, relevant_content
):
    import io
    import tarfile

    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("../escaped.tex")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    handler = relevant_content.ArxivHandler("https://arxiv.org/abs/1234.5678")
    monkeypatch.setattr(handler, "fetch_tex", lambda: str(archive))
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    monkeypatch.setattr(relevant_content.tempfile, "mkdtemp", lambda: str(extract_dir))
    assert handler.extract_text() == ""  # extraction refused; the handler falls back to ""
    assert not (tmp_path / "escaped.tex").exists()


def test_kaggle_resources_are_scraped_from_the_given_url(relevant_content):
    url = "https://www.kaggle.com/competitions/x/discussion/1"
    # used the script's global loop variable `link`: NameError when imported
    assert relevant_content.process_resource(url) == f"posts of {url}"


def test_hyperparameter_search_runs_are_graded_with_their_code(tmp_path):
    from xgboost import XGBRegressor

    data = tmp_path / "data"
    competition = _prepared(data, HPS_ID, OPTIVER_ANSWERS)
    competition.sample_submission.parent.mkdir(parents=True)  # the score is measured over it
    competition.sample_submission.write_text("row_id,target\n0-1,0.0\n0-2,0.0\n")
    run = tmp_path / "run"
    folder = run / "submission"
    folder.mkdir(parents=True)
    (folder / "baseline_submission.csv").write_text("row_id,target\n0-1,0.003\n0-2,0.004\n")
    (folder / "submission.csv").write_bytes(OPTIVER_ANSWERS)
    (folder / "baseline_model.pkl").write_bytes(pickle.dumps(XGBRegressor()))  # as described
    (folder / "model.pkl").write_bytes(pickle.dumps(XGBRegressor(max_depth=3)))
    # the JSONL line make_submission.py writes for this run
    line = {
        "competition_id": HPS_ID,
        "submission_path": str(folder),
        "solution_file_path": str(run / "code"),
    }
    jsonl = _jsonl(tmp_path / "s.jsonl", [line])

    def graded() -> dict:
        out = tmp_path / f"out{len(list(tmp_path.glob('out*')))}"
        grade_jsonl(jsonl, out, Registry(data))
        (report,) = json.loads(next(out.iterdir()).read_text())["competition_reports"]
        return report

    assert not graded()["valid_submission"]  # no solution code
    (run / "code").mkdir()
    (run / "code" / "search.py").write_text("import optuna\noptuna.create_study().optimize(f)\n")
    report = graded()
    assert report["valid_submission"] and float(report["score"]) > 0


def test_evaluation_splits_list_only_gradeable_challenges():
    import inspect

    registry = Registry()
    for split in sorted(registry.get_splits_dir().glob("*.txt")):
        for competition_id in split.read_text().split():
            grade_fn = registry.get_competition(competition_id).grader.grade_fn
            source = inspect.getsource(grade_fn)
            # the CSDI graders are stubs, and all.txt listed both CSDI challenges
            assert "raise NotImplementedError" not in source, (split.name, competition_id)


def test_preparing_an_uncategorised_competition_says_why_it_fails(tmp_path):
    import dataclasses

    from timeseriesgym.data import download_and_prepare_dataset

    listed = Registry(tmp_path).get_competition("spaceship-titanic")
    unlisted = dataclasses.replace(listed, id="new-challenge")
    with pytest.raises(ValueError, match="constants.py"):  # was UnboundLocalError: zipfile
        download_and_prepare_dataset(unlisted, skip_leaderboard=True)


def test_competition_with_own_grader_can_inherit_the_description(tmp_path, monkeypatch):
    # fixed by 4b9de66 (parent_id was only bound where the grader is inherited); no test then
    competitions = tmp_path / "competitions"
    for competition_id in ("parent", "child"):
        (competitions / competition_id).mkdir(parents=True)
    (competitions / "parent" / "description.md").write_text("parent description")
    grader = "grader:\n  name: g\n  grade_fn: timeseriesgym.metrics:root_mean_squared_log_error\n"
    common = "competition_type: simple\ndataset:\n  answers: a.csv\n  sample_submission: s.csv\n"
    (competitions / "parent" / "config.yaml").write_text(
        f"id: parent\nname: P\ndescription: x.md\n{common}{grader}"
    )
    (competitions / "child" / "config.yaml").write_text(
        f"id: child\nname: C\nparent_id: parent\n{common}{grader}"
    )
    registry = Registry(tmp_path / "data")
    monkeypatch.setattr(registry, "get_competitions_dir", lambda: competitions)
    assert registry.get_competition("child").description == "parent description"


# ---------------------------------------------------------------- HDF5 submissions


def _h5_bytes(tmp_path: Path, name: str, datasets: dict) -> bytes:
    import h5py

    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as f:
        for key, data in datasets.items():
            f.create_dataset(key, data=data)
    return path.read_bytes()


@pytest.mark.parametrize(
    "how", ["external", "soft", "dangling", "virtual", "external_storage", "datatype", "cycle"]
)
def test_h5_submissions_must_be_self_contained(tmp_path, how):
    import h5py

    answers = tmp_path / "answers.h5"  # stands in for the private answers
    answers.write_bytes(_h5_bytes(tmp_path, "a", {"labels": np.arange(20.0)}))
    raw = tmp_path / "raw.bin"
    np.arange(20.0).tofile(raw)
    path = tmp_path / "submission.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("submission")
        group.create_dataset("easy", data=np.ones(10))
        if how == "external":  # used to be read, as root in the grading server
            group["hard"] = h5py.ExternalLink(str(answers), "labels")
        elif how == "soft":
            f.create_dataset("elsewhere", data=np.ones(20))
            group["hard"] = h5py.SoftLink("/elsewhere")
        elif how == "dangling":
            group["hard"] = h5py.SoftLink("/does/not/exist")
        elif how == "virtual":
            layout = h5py.VirtualLayout(shape=(20,), dtype="f8")
            layout[:] = h5py.VirtualSource(str(answers), "labels", shape=(20,))
            group.create_virtual_dataset("hard", layout)
        elif how == "external_storage":
            group.create_dataset("hard", shape=(20,), dtype="f8", external=[(str(raw), 0, 160)])
        elif how == "datatype":
            group["hard"] = np.dtype("f8")  # a committed datatype, not a dataset
        else:
            group["hard"] = group  # hard-link cycle: load_arbitrary_h5 never returned
    with pytest.raises(InvalidSubmissionError, match="Link|virtual|groups and datasets|deeper"):
        load_submissions(path)


def test_self_contained_h5_submissions_load(tmp_path):
    import h5py

    path = tmp_path / "submission.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("submission/easy", data=np.ones(10))
        f.create_dataset("submission/nested/hard", data=np.zeros(3))
    loaded = load_submissions(path)
    assert np.array_equal(loaded["easy"], np.ones(10))
    assert np.array_equal(loaded["nested"]["hard"], np.zeros(3))


def test_validation_does_not_follow_h5_links_to_the_answers(tmp_path):
    import h5py

    labels = np.zeros(50)
    labels[20:25] = 1
    competition = _prepared(
        tmp_path / "data", "ucr-anomaly-detection", _h5_bytes(tmp_path, "a", {"labels/x": labels})
    )
    submission = tmp_path / "submission.h5"
    with h5py.File(submission, "w") as f:
        f["submission"] = h5py.ExternalLink(str(competition.answers), "labels")
    valid, message = validate_submission(submission, competition)
    assert not valid and "ExternalLink" in message  # was valid: the answers graded themselves
