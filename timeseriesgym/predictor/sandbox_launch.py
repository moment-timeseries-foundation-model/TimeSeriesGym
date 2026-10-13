"""The grader's default launcher: the predictor runs in the grading sandbox.

SandboxLauncher implements the launcher interface of launch.py with the interactive grading
sandbox (timeseriesgym/sandbox.py: interactive): the predictor runs as the user `sandbox` when
grading runs as root (the grading container), otherwise as the grading user confined by
Landlock, with a copy of the predictor directory, a scrubbed environment, no sockets, limits on
time and memory, and no read access to the challenge's private files (`job.protected`: the full
data and the answers). If the sandbox could read them, grading stops with a SandboxError, as for
code challenges. The plain subprocess of launch.py (SubprocessLauncher) runs the predictor with
the grading user's access, answers included: `timeseriesgym grade-predictor --trusted` uses it,
for trusted predictors only.

The predictor runs with the Python of $TSGYM_PREDICTOR_PYTHON, which the grader image sets to a
copy of the agent environment (environment/Dockerfile), so that it meets the packages it was
written with; without it, with the grader's own Python.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass

from .launch import Launcher, PredictorJob, PredictorProcess

PYTHON_ENV = "TSGYM_PREDICTOR_PYTHON"
ENTRYPOINT = "timeseriesgym.predictor.runner:serve"
GRACE_SEC = 60  # past the grader's own deadline, before the sandbox's watchdog stops it


def predictor_python() -> str:
    """The interpreter that runs predictors: $TSGYM_PREDICTOR_PYTHON, else this one."""
    return os.environ.get(PYTHON_ENV) or sys.executable


@dataclass
class SandboxLauncher:
    """Runs the predictor in the interactive grading sandbox (see the module docstring), with
    the interpreter `python` (default: predictor_python())."""

    python: str | None = None

    @contextlib.contextmanager
    def __call__(self, job: PredictorJob) -> Iterator[PredictorProcess]:
        from timeseriesgym import sandbox

        limits = sandbox.Limits(
            timeout_sec=math.ceil(job.timeout_sec) + GRACE_SEC,
            memory_bytes=job.memory_bytes or sandbox.memory_limit(),
        )
        with sandbox.interactive(
            ENTRYPOINT,
            {"predictor": "predictor"},
            limits,
            agent_dirs={"predictor": job.predictor_dir},
            protected=job.protected,
            python=self.python or predictor_python(),
            env={"PYTHONDONTWRITEBYTECODE": "1", **job.env},
        ) as session:
            yield session


def default_launcher() -> Launcher:
    """The launcher of `timeseriesgym grade` and `grade-predictor`: the grading sandbox."""
    return SandboxLauncher()
