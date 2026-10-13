"""Agent code runs in the grading sandbox (timeseriesgym/sandbox.py), never in the
grading process, which holds the answers.

The parity tests also pass on the commit before the sandbox: well-behaved submissions score as
the native graders scored them in-process. The other tests fail there: in the grading process,
agent code could read the answers and the grader's environment, write files, leave processes
behind, and crash, exhaust or hang the grader.
"""

import io
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from timeseriesgym import sandbox
from timeseriesgym.grade import grade_code
from timeseriesgym.registry import Registry

REPO = Path(__file__).resolve().parents[2]
STOMP = "stomp-R-to-python"
RESNET = "resnet-tensorflow-to-pytorch"
STOMP_COLUMNS = [
    *(f"mp_self_{c}" for c in ("mp", "pi", "rmp", "rpi", "lmp", "lpi")),
    "mp_join_mp",
    "mp_join_pi",
]

# Not STOMP: a function of the inputs with the outputs' types and shapes, and whose outputs the
# test answers are.
STOMP_SUBMISSION = """
import numpy as np


def stomp(ref_data, query_data, window_size, exclusion_zone=0.5, verbose=2):
    x = np.asarray(ref_data, dtype=float)
    n = len(x) - window_size + 1
    mp = np.array([x[i : i + window_size].std() for i in range(n)])
    pi = np.arange(1, n + 1, dtype=float)
    if query_data is None:
        return mp, pi, mp + 1, pi + 1, np.full(n, np.inf), np.full(n, -np.inf)
    return mp * float(np.mean(query_data)), pi, None, None, None, None
"""


def _stomp_competition(tmp_path: Path):
    """A prepared stomp-R-to-python whose answers are STOMP_SUBMISSION's outputs."""
    competition = Registry(tmp_path / "data").get_competition(STOMP)
    competition.public_dir.mkdir(parents=True)
    competition.private_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    ref, query = rng.normal(size=550), rng.normal(size=550)
    np.save(competition.public_dir / "ref_data.npy", ref)
    np.save(competition.public_dir / "query_data.npy", query)
    namespace = {}
    exec(STOMP_SUBMISSION, namespace)
    self_join = namespace["stomp"](ref, None, 30)
    join = namespace["stomp"](ref, query, 30)
    answers = np.column_stack([*self_join, *join[:2]])
    pd.DataFrame(answers, columns=STOMP_COLUMNS).to_csv(competition.answers, index=False)
    return competition


def _grade_stomp(tmp_path: Path, code: str, competition=None):
    """The report for `code` as submission.py, with ANSWERS replaced by the answers' path."""
    competition = competition or _stomp_competition(tmp_path)
    submission = tmp_path / "submission.py"
    submission.write_text(code.replace("ANSWERS", repr(str(competition.answers))))
    return grade_code(submission, competition)


def _stomp_scores(report) -> tuple:
    return (
        report.valid_submission,
        report.defined_functions,
        report.executed_functions,
        report.test_metric,
    )


# ---------------------------------------------------------------- parity


STOMP_VARIANTS = {
    "as is": ("", ("1/1", "1/1", "2/2")),
    "lists": ("out = tuple(o if o is None else list(o) for o in out)", ("1/1", "0/1", "2/2")),
    "objects": (
        "out = tuple(o if o is None else o.astype(object) for o in out)",
        ("1/1", "1/1", "2/2"),
    ),
    "wrong": ("out = (out[0] + 0.5, *out[1:])", ("1/1", "1/1", "0/2")),
    "join not None": (
        "out = out if query_data is None else (*out[:2],) * 3",
        ("1/1", "1/1", "1/2"),
    ),
    "five": ("out = out[:5]", ("1/1", "0/1", "0/2")),
    "a None": ("out = (None, *out[1:])", ("1/1", "0/1", "0/2")),
    "strings": ("out = (out[0].astype(str), *out[1:])", ("1/1", "1/1", "0/2")),
    "raises": ("raise ValueError('no')", ("1/1", "0/1", "0/2")),
}


@pytest.mark.parametrize("variant", STOMP_VARIANTS)
def test_stomp_scores_as_the_native_grader(tmp_path, variant):
    change, expected = STOMP_VARIANTS[variant]
    code = STOMP_SUBMISSION.replace("def stomp(", "def _stomp(") + (
        "\n\ndef stomp(ref_data, query_data, window_size, exclusion_zone=0.5, verbose=2):\n"
        "    out = _stomp(ref_data, query_data, window_size, exclusion_zone, verbose)\n"
        f"    {change}\n"
        "    return out\n"
    )
    assert _stomp_scores(_grade_stomp(tmp_path, code)) == (True, *expected)


@pytest.mark.parametrize("code", ["def stomp(:\n", "import a_module_that_does_not_exist\n"])
def test_stomp_submissions_that_do_not_import_are_invalid(tmp_path, code):
    assert _stomp_scores(_grade_stomp(tmp_path, code)) == (False, None, None, None)


RESNET_SUBMISSION = """
import numpy as np
import torch
from torch import nn


class ResNet(nn.Module):
    def __init__(
        self, output_directory, input_shape, n_classes, n_feature_maps, verbose=False,
        random_seed=42,
    ):
        super().__init__()
        torch.manual_seed(random_seed)
        self.linear = nn.Linear(input_shape[0] * input_shape[1], n_classes)

    def forward(self, x):
        return self.linear(x.flatten(1))


def train(
    model, train_data, val_data, batch_size=64, n_epochs=1, learning_rate=0.001, verbose=False
):
    return model


def predict(model, test_data):
    x, y = test_data.tensors
    mean = x.mean((1, 2)).numpy()
    return y.numpy(), np.stack([-mean, mean], axis=1)
"""


def _resnet_competition(tmp_path: Path, wrong: int):
    """A prepared resnet-tensorflow-to-pytorch where RESNET_SUBMISSION's predictions are right
    except on `wrong` of the 100 test rows."""
    competition = Registry(tmp_path / "data").get_competition(RESNET)
    public = competition.public_dir
    public.mkdir(parents=True)
    competition.private_dir.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)

    def dataset(n):
        x = torch.randn(n, 1, 50, generator=generator)
        return torch.utils.data.TensorDataset(x, (x.mean((1, 2)) > 0).long())

    torch.save(torch.randn(10, 1, 50, generator=generator), public / "x.pt")
    torch.save(dataset(64), public / "train_data.pt")
    torch.save(dataset(32), public / "val_data.pt")
    x, labels = dataset(100).tensors
    labels = labels.numpy().copy()
    labels[:wrong] = 1 - labels[:wrong]
    zeros = torch.zeros(len(x), dtype=torch.long)  # the dummy labels the native data hold
    torch.save(torch.utils.data.TensorDataset(x, zeros), public / "test_data.pt")
    pd.DataFrame({"label": labels}).to_csv(competition.answers, index=False)
    return competition


RESNET_VARIANTS = {
    "right": (0, "", ("1/1", "1/1", "1/1", "1/1", "2/2", "2/2", "1/1")),
    "accuracy 0.8": (20, "", ("1/1", "1/1", "1/1", "1/1", "2/2", "2/2", "0/1")),  # not > 0.8
    "accuracy 0.81": (19, "", ("1/1", "1/1", "1/1", "1/1", "2/2", "2/2", "1/1")),
    "no predictions": (
        0,
        "predict = lambda model, test_data: None",
        ("1/1", "1/1", "1/1", "1/1", "2/2", "1/2", "0/1"),
    ),
    "1-d scores": (  # argmax over axis 1 fails
        0,
        "_p = predict\npredict = lambda model, test_data: (_p(model, test_data)[0], np.zeros(100))",
        ("1/1", "1/1", "1/1", "1/1", "2/2", "1/2", "0/1"),
    ),
    "3-d scores": (  # (100, 2) predictions do not broadcast against the labels
        0,
        "predict = lambda model, test_data: (None, np.zeros((100, 3, 2)))",
        ("1/1", "1/1", "1/1", "1/1", "2/2", "1/2", "0/1"),
    ),
    "no class": (0, "del ResNet", ("0/1", "0/1", "0/1", "0/1", "2/2", "1/2", "0/1")),
}


@pytest.mark.parametrize("variant", RESNET_VARIANTS)
def test_resnet_scores_as_the_native_grader(tmp_path, variant):
    wrong, change, expected = RESNET_VARIANTS[variant]
    competition = _resnet_competition(tmp_path, wrong)
    submission = tmp_path / "submission.py"
    submission.write_text(RESNET_SUBMISSION + change + "\n")
    report = grade_code(submission, competition)
    assert report.valid_submission
    assert (
        report.defined_classes,
        report.initialized_classes,
        report.defined_class_methods,
        report.executed_class_methods,
        report.defined_functions,
        report.executed_functions,
        report.test_metric,
    ) == expected


# ---------------------------------------------------------------- what agent code cannot do


def test_agent_code_cannot_read_the_answers(tmp_path):
    cheat = """
import numpy as np


def stomp(ref_data, query_data, window_size, exclusion_zone=0.5, verbose=2):
    try:
        answers = np.loadtxt(ANSWERS, delimiter=",", skiprows=1)
    except OSError:
        answers = np.zeros((521, 8))
    if query_data is None:
        return tuple(answers[:, i] for i in range(6))
    return answers[:, 6], answers[:, 7], None, None, None, None
"""
    assert _grade_stomp(tmp_path, cheat).test_metric == "0/2"  # was 2/2, read from the answers


def test_agent_code_does_not_see_the_graders_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_KEY", "a credential")
    code = STOMP_SUBMISSION + "\nimport os\nif 'KAGGLE_KEY' in os.environ:\n    stomp = None\n"
    assert _grade_stomp(tmp_path, code).test_metric == "2/2"  # was 0/2: the key was visible


def test_agent_code_cannot_write_outside_its_work_directory(tmp_path):
    marker = tmp_path / "written-by-agent-code"
    code = f"open({str(marker)!r}, 'w').close()\n" + STOMP_SUBMISSION
    report = _grade_stomp(tmp_path, code)
    assert not marker.exists()
    assert not report.valid_submission  # PermissionError at import


def _processes_with(tag: str) -> list[int]:
    found = []
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            try:
                cmdline = Path(entry.path, "cmdline").read_bytes()
            except OSError:
                continue
            if tag.encode() in cmdline:
                found.append(int(entry.name))
    return found


def test_processes_started_by_agent_code_do_not_survive_grading(tmp_path):
    tag = f"tsgym-sandbox-test-{uuid.uuid4().hex}"
    # A daemon that leaves the session twice over (setsid, then a double fork), so that it is
    # neither in the sandbox's process group nor its child.
    code = (
        textwrap.dedent(f"""
        import os, subprocess, sys
        if os.fork() == 0:
            os.setsid()
            if os.fork() == 0:
                subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", {tag!r}])
            os._exit(0)
        os.wait()
        """)
        + STOMP_SUBMISSION
    )
    try:
        report = _grade_stomp(tmp_path, code)
        time.sleep(0.5)
        assert report.test_metric == "2/2"
        assert not _processes_with(tag)  # it outlived the grading
    finally:
        for pid in _processes_with(tag):
            os.kill(pid, signal.SIGKILL)


def test_agent_code_that_hangs_times_out(tmp_path, monkeypatch):
    monkeypatch.setenv(sandbox.TIMEOUT_ENV, "10")
    start = time.monotonic()
    assert not _grade_stomp(tmp_path, "while True:\n    pass\n").valid_submission  # hung
    assert time.monotonic() - start < 60


def test_agent_code_cannot_take_the_graders_memory(tmp_path, monkeypatch):
    monkeypatch.setenv(sandbox.MEMORY_ENV, "1024")
    code = "_hoard = b'x' * 2 * 2**30\n" + STOMP_SUBMISSION
    assert not _grade_stomp(tmp_path, code).valid_submission  # was valid: no limit


@pytest.mark.parametrize(
    "code",
    [
        "import ctypes\nctypes.string_at(0)\n",  # a segmentation fault ended the grader
        "import os\nos._exit(0)\n",  # so did this
        "raise SystemExit(0)\n",  # SystemExit went through the grader's except clause
    ],
)
def test_agent_code_that_ends_its_process_is_invalid(tmp_path, code):
    assert not _grade_stomp(tmp_path, code + STOMP_SUBMISSION).valid_submission


def test_forged_sandbox_outputs_are_invalid(tmp_path):
    # Agent code runs in the same process as the sandbox half: it can replace what that writes.
    forge = """
import os
from timeseriesgym import sandbox


def _write_json(work, name, value):
    os.makedirs(work / "out", exist_ok=True)
    os.symlink(ANSWERS, work / "out" / name)


sandbox.write_json = _write_json
"""
    assert not _grade_stomp(tmp_path, forge + STOMP_SUBMISSION).valid_submission


def test_grading_stops_if_the_sandbox_could_read_the_answers(tmp_path, monkeypatch):
    shared = Path(tempfile.mkdtemp())  # the answers are readable for every user here
    try:
        os.chmod(shared, 0o755)
        competition = _stomp_competition(shared)
        monkeypatch.setattr(sandbox, "LANDLOCK", False)  # as on a kernel without Landlock
        with pytest.raises(sandbox.SandboxError, match="read the challenge's private files"):
            _grade_stomp(tmp_path, STOMP_SUBMISSION, competition)
        monkeypatch.setenv(sandbox.UNCONFINED_ENV, "1")  # to grade trusted submissions anyway
        assert _grade_stomp(tmp_path, STOMP_SUBMISSION, competition).test_metric == "2/2"
    finally:
        shutil.rmtree(shared)


@pytest.mark.skipif(
    os.geteuid() != 0 or sandbox.sandbox_user() is None,
    reason="needs root and a sandbox user, to grade as an unprivileged user",
)
def test_without_root_landlock_keeps_the_answers_from_agent_code():
    # Grading as a normal user, with nobody to switch to: the sandbox has the grader's uid, so
    # file permissions let it read the answers, and only Landlock keeps it from them.
    uid, gid = sandbox.sandbox_user()
    home = Path(tempfile.mkdtemp())
    try:
        script = home / "grade.py"
        script.write_text(
            textwrap.dedent(f"""
            import sys
            from pathlib import Path
            sys.path.insert(0, {str(REPO)!r})
            from tests.unit.test_sandbox import _grade_stomp, _stomp_scores
            print(_stomp_scores(_grade_stomp(Path({str(home)!r}), sys.stdin.read())))
            """)
        )
        os.chown(home, uid, gid)
        cheat = "import numpy\nnumpy.loadtxt(ANSWERS, delimiter=',', skiprows=1)\n"
        out = subprocess.run(
            [sys.executable, str(script)],
            input=cheat + STOMP_SUBMISSION,
            user=uid,
            group=gid,
            extra_groups=[],
            env={**os.environ, "HOME": str(home)},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert out.returncode == 0, out.stderr[-3000:]
        # was (True, "1/1", "1/1", "2/2"): the answers were read
        assert out.stdout.strip().splitlines()[-1] == "(False, None, None, None)"
    finally:
        shutil.rmtree(home)


# ---------------------------------------------------------------- reading sandbox outputs


def test_sandbox_outputs_are_read_without_following_links(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.json").write_text('{"a": 1}')
    assert sandbox.read_json(tmp_path, "a.json") == {"a": 1}
    assert sandbox.read_json(tmp_path, "absent.json") is None
    (tmp_path / "secret").write_text('{"answers": 1}')
    (out / "link.json").symlink_to(tmp_path / "secret")
    os.mkfifo(out / "fifo.json")
    (out / "big.json").write_text("[" + "0," * sandbox.MAX_JSON_BYTES + "0]")
    (out / "deep.json").write_text("[" * 100 + "]" * 100)
    (out / "bad.json").write_bytes(b"\xff")
    for name in ("link.json", "fifo.json", "big.json", "deep.json", "bad.json"):
        with pytest.raises(ValueError):
            sandbox.read_json(tmp_path, name)
    shutil.move(out, tmp_path / "real-out")
    (tmp_path / "out").symlink_to(tmp_path / "real-out")
    with pytest.raises(ValueError, match="output directory"):
        sandbox.read_json(tmp_path, "a.json")


def test_sandbox_arrays_must_be_plain_numbers(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "a.npy", np.arange(3.0))
    np.save(out / "c.npy", np.arange(3) + 1j)
    np.save(out / "o.npy", np.array([None, 1], dtype=object), allow_pickle=True)
    np.save(out / "s.npy", np.array(["a"]))
    header = io.BytesIO()
    shape = {"descr": "<f8", "fortran_order": False, "shape": (2**60,)}
    np.lib.format.write_array_header_1_0(header, shape)
    (out / "huge.npy").write_bytes(header.getvalue())  # declares 8 EiB and holds none
    assert sandbox.read_npy(tmp_path, "a.npy").tolist() == [0.0, 1.0, 2.0]
    assert sandbox.read_npy(tmp_path, "c.npy").dtype == np.complex128
    for name in ("o.npy", "s.npy", "huge.npy"):
        with pytest.raises(ValueError):
            sandbox.read_npy(tmp_path, name)
