"""Grading of the predictor challenges (gift-eval-short-horizon,
time-series-library-long-horizon-forecast).

The submission is a directory with predict.py (contract.py). The grader builds the test windows
from the full private data exactly as the prepare built the answers (windows.py: same windows,
same order), runs the predictor on them through a launcher (launch.py), in rounds that never
show a test value before its forecast is committed (protocol.py), assembles the forecasts in
the challenge's submission format and scores them with the challenge's unchanged grade function
(score_forecasts, the only place the metric is called).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from timeseriesgym.grade_helpers import GradingInput
from timeseriesgym.utils import InvalidSubmissionError, get_logger, load_answers

from . import contract, protocol
from . import windows as pw
from .launch import Launcher, PredictorJob, default_launcher, run_job, thread_env

logger = get_logger(__name__)

DEFAULTS = {
    "timeout_sec": 2400,  # the whole predictor run, importing predict.py included
    "cpus": 8,
    "memory_mb": None,  # address-space limit of the predictor, if the launcher applies one
    "max_bytes": contract.MAX_BYTES,
    "max_files": contract.MAX_ENTRIES,
    "max_batch_windows": 1024,
    "max_batch_values": 32 * 1024**2,
}


def settings(competition) -> dict:
    """The competition's `predictor_config` over the defaults."""
    return {**DEFAULTS, **(competition.predictor_config or {})}


def test_windows(competition, with_labels: bool = False) -> pw.WindowSet:
    """The hidden test windows, built from the full data in the private directory."""
    return pw.build(
        settings(competition)["windows"], competition.private_dir, "test", with_labels=with_labels
    )


def run_predictor(
    predictor: Path, ws: pw.WindowSet, config: dict, launcher: Launcher, protected=()
) -> tuple[dict[str, np.ndarray], dict]:
    """Forecasts of the predictor in `predictor` (a checked copy) for every window of `ws`.
    `protected`: the paths the predictor must not read (for a sandboxing launcher). Raises
    InvalidSubmissionError if it fails, breaks the protocol or runs out of time."""
    timeout = float(config["timeout_sec"])
    memory = config.get("memory_mb")
    job = PredictorJob(
        predictor_dir=predictor,
        timeout_sec=timeout,
        cpus=int(config["cpus"]),
        memory_bytes=int(memory) * 1024**2 if memory else None,
        env=thread_env(int(config["cpus"])),
        protected=[Path(p) for p in protected],
    )
    arrays = ws.empty()
    try:
        stats = run_job(
            launcher,
            job,
            ws.windows,
            ws.sink(arrays),
            max_windows=int(config["max_batch_windows"]),
            max_values=int(config["max_batch_values"]),
        )
    except protocol.PredictorError as e:
        raise InvalidSubmissionError(f"predictor: {e}") from None
    for name, a in arrays.items():  # every slot filled once: a protocol invariant
        if not np.isfinite(a).all():
            raise RuntimeError(f"{name}: forecasts missing after a complete run")
    stats["limit_s"] = timeout
    return arrays, stats


def score_forecasts(competition, ws: pw.WindowSet, forecasts: dict[str, np.ndarray]):
    """The score of the forecasts: assembled in the challenge's submission format and scored by
    the challenge's grade function against its answers, as a static submission would be (None
    if the grader rejects them). The only place the metric is called; a metric that needs more
    than the answers (e.g. a scale computed from each window's context) gets it from `ws`."""
    grading_input = GradingInput(
        submission=ws.to_native(forecasts), answers=load_answers(competition.answers)
    )
    return competition.grader(grading_input)


def label_arrays(ws: pw.WindowSet, answers) -> dict[str, np.ndarray]:
    """The answers' label arrays, per dataset: TSL a dict of arrays; GIFT-Eval a one-row
    compound array, or a group whose `target` is one (answers that also hold a metric's
    per-window scales)."""
    if ws.challenge == "tsl":
        return {d.name: np.asarray(answers[d.name]) for d in ws.datasets}
    if isinstance(answers, dict):
        answers = answers["target"]
    return {d.name: np.asarray(answers[d.name][0]) for d in ws.datasets}


def check_windows(ws: pw.WindowSet, answers) -> None:
    """The windows' labels are the answers, window for window (same windows, same order).
    Raises AssertionError otherwise."""
    for d in ws.datasets:
        ref = label_arrays(ws, answers)[d.name]
        got = d.labels.astype(ref.dtype)
        same = got.shape == ref.shape and np.array_equal(got, ref, equal_nan=True)
        assert same, f"{d.name}: the test windows' labels differ from the answers"


def grade_predictor_dir(
    submission: Path, competition, launcher: Launcher | None = None
) -> tuple[object, dict]:
    """The score of the predictor directory `submission`, and run statistics. Raises
    InvalidSubmissionError if it has the wrong format or fails."""
    config = settings(competition)
    max_bytes, max_files = int(config["max_bytes"]), int(config["max_files"])
    with tempfile.TemporaryDirectory(prefix="tsgym-predictor-", ignore_cleanup_errors=True) as tmp:
        staged = Path(tmp) / "predictor"
        try:  # the grader checks and runs its own copy, which the agent cannot change
            contract.copy_predictor(Path(submission), staged, max_bytes, max_files)
            contract.check_predictor(staged, max_bytes, max_files)
        except (contract.ContractError, OSError) as e:
            raise InvalidSubmissionError(f"predictor directory {submission}: {e}") from None
        started = time.monotonic()
        ws = test_windows(competition)
        built = time.monotonic() - started
        protected = [competition.private_dir, competition.raw_dir]
        launcher = launcher or default_launcher()
        forecasts, stats = run_predictor(staged, ws, config, launcher, protected)
    stats["windows_built_s"] = round(built, 3)
    score = score_forecasts(competition, ws, forecasts)
    logger.info(f"{competition.id}: predictor run {stats}")
    return score, stats
