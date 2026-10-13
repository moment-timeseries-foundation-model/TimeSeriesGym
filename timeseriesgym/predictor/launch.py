"""How the grader starts a predictor process: the interface a sandbox implements, and a plain
subprocess launcher to use until one does.

The grader (grading.py) needs one thing from a launcher: a running `python -I runner.py
<predictor dir>` whose stdin and stdout are pipes to the grader (the protocol, protocol.py),
whose stderr is kept for error messages, and which is gone, with everything it started, when
the grader is done with it. Everything else (which user it runs as, what it can read, network,
memory) is the launcher's business, so a sandbox can be dropped in without touching the
protocol or the grading:

    class MySandbox:
        @contextlib.contextmanager
        def __call__(self, job: PredictorJob) -> Iterator[PredictorProcess]:
            ...  # copy job.predictor_dir in, start job.argv(python) inside, yield its pipes
            ...  # finally: kill everything that was started

    grade_predictor(path, competition, launcher=MySandbox())

The grader enforces the wall-clock limit itself (every read and write of the protocol has a
deadline) and kills nothing: leaving the context must stop the process. A sandbox that runs
`module:function(work_dir, job)` entry points can run `timeseriesgym.predictor.runner:serve`
instead of runner.py, provided that the entry point gets the grader's pipes as fds 0 and 1;
`job.protected` lists what it must keep unreadable.

SubprocessLauncher is NOT a sandbox. It runs the predictor as the grading user, so the
predictor can read and write whatever that user can, including the answers. Use it for trusted
predictors (tests, the agent's own check, baselines) or inside a disposable container that
holds no more than the grading needs.
"""

from __future__ import annotations

import contextlib
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol

from . import protocol

RUNNER = Path(__file__).with_name("runner.py")


@dataclass
class PredictorJob:
    """What to run. `predictor_dir` is the grader's private, checked copy of the submission
    (the launcher may copy it again, e.g. into a sandbox); `runner` imports only the standard
    library and numpy (a sandbox that calls `function(work_dir, job)` can run
    `timeseriesgym.predictor.runner:serve` instead). `timeout_sec` is the wall-clock limit of
    the whole run, which the grader enforces; a launcher may also use it, e.g. for a CPU-time
    limit. `env` holds variables the predictor should see (thread counts). `protected` are the
    challenge's private paths (answers, full data), which a sandbox must keep unreadable."""

    predictor_dir: Path
    runner: Path = RUNNER
    timeout_sec: float = 2400
    cpus: int = 1
    memory_bytes: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    protected: list[Path] = field(default_factory=list)

    def argv(self, python: str, predictor_dir: Path | None = None) -> list[str]:
        """The command line, for the given interpreter (and copy of the predictor)."""
        return [python, "-I", str(self.runner), str(predictor_dir or self.predictor_dir)]


class PredictorProcess(Protocol):
    """A running predictor, as the grader sees it."""

    stdin: IO[bytes]  # the grader writes requests here
    stdout: IO[bytes]  # and reads replies from here

    def stderr_tail(self, limit: int = 3000) -> str:
        """The end of what the predictor wrote to stderr (and printed: see runner.py)."""


class Launcher(Protocol):
    def __call__(self, job: PredictorJob) -> contextlib.AbstractContextManager[PredictorProcess]:
        """Start the job; on leaving the context, the process and every process it started
        must be gone."""


def thread_env(cpus: int) -> dict[str, str]:
    """Thread counts for the usual numeric libraries."""
    n = str(int(cpus))
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    return dict.fromkeys(names, n)


def _tail(f, limit: int) -> str:
    f.flush()
    size = f.seek(0, os.SEEK_END)
    f.seek(max(0, size - limit))
    return f.read().decode(errors="replace")


@dataclass
class _Process:
    proc: subprocess.Popen
    log: IO[bytes]

    @property
    def stdin(self) -> IO[bytes]:
        return self.proc.stdin

    @property
    def stdout(self) -> IO[bytes]:
        return self.proc.stdout

    def stderr_tail(self, limit: int = 3000) -> str:
        return _tail(self.log, limit)


@dataclass
class SubprocessLauncher:
    """Runs the predictor as a child process of the grader: in its own session (so that it and
    its children can be killed together), with a minimal environment, the predictor directory
    as its working directory, and CPU-time and (if the job sets it) address-space limits. Not
    a sandbox (see the module docstring): `job.protected` stays readable. `python`: the
    interpreter with the predictor's packages (default: this one)."""

    python: str = sys.executable

    @contextlib.contextmanager
    def __call__(self, job: PredictorJob) -> Iterator[PredictorProcess]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(job.predictor_dir),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            **job.env,
        }
        cpu_sec = int(job.timeout_sec * max(1, job.cpus)) + 60

        def limits() -> None:  # in the child, before exec
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
            if job.memory_bytes:
                resource.setrlimit(resource.RLIMIT_AS, (job.memory_bytes, job.memory_bytes))

        with tempfile.TemporaryFile(prefix="predictor-stderr-") as log:
            proc = subprocess.Popen(
                job.argv(self.python),
                cwd=job.predictor_dir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                bufsize=0,
                start_new_session=True,
                preexec_fn=limits,
            )
            try:
                yield _Process(proc, log)
            finally:
                for f in (proc.stdin, proc.stdout):
                    with contextlib.suppress(OSError):
                        f.close()
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)  # the predictor and its children
                proc.wait()


def run_job(launcher: Launcher, job: PredictorJob, windows: list, sink, **kw) -> dict:
    """Start the job and run the protocol (protocol.run, with its keyword arguments) on
    `windows`, passing each forecast to `sink(window, array)`. Returns the run statistics.
    Raises protocol.PredictorError, with the end of the predictor's stderr."""
    with launcher(job) as proc:
        channel = protocol.Channel(proc.stdin, proc.stdout, time.monotonic() + job.timeout_sec)
        try:
            return protocol.run(channel, windows, sink, **kw)
        except protocol.PredictorError as e:
            time.sleep(0.2)  # let the predictor finish writing its traceback
            limit = f"limit {job.timeout_sec:g} s"
            raise protocol.PredictorError(f"{e} ({limit}); stderr: {proc.stderr_tail()}") from None


def default_launcher() -> Launcher:
    """The launcher of `timeseriesgym grade`: a SubprocessLauncher with the interpreter named
    by $TSGYM_PREDICTOR_PYTHON (the agent environment's Python, so that the predictor meets the
    packages it was developed with), else this one."""
    return SubprocessLauncher(os.environ.get("TSGYM_PREDICTOR_PYTHON") or sys.executable)
