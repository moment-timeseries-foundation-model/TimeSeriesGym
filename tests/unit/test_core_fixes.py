"""Regression tests for fixes in the core package, the container environment and the runners."""

import importlib
import os
import pickle
import re
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from timeseriesgym.grade import validate_submission
from timeseriesgym.registry import Registry
from timeseriesgym.utils import (
    InvalidSubmissionError,
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
