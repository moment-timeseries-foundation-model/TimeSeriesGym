"""The grading step of a run, after its agent's container has stopped.

The agent's container has only the public data (agents/run.py). Its outputs are extracted into
the run directory, and this step grades them in a separate container of the grader image
(environment/Dockerfile, target `grader`). That container is the only one with the private data:
the competition's prepared `private` and `public` directories, read-only. It has no network
and runs `timeseriesgym grade` on the JSONL line that experiments/make_submission.py writes for
the run, so the run is graded as in the manual workflow. The report and the grader's log are
written to `<run_dir>/grading/`.

The grader runs as root, so that the grading sandbox (timeseriesgym/sandbox.py) runs agent code
(code submissions, models, predictors) as the image's user `sandbox`, which cannot read the
private data. What it writes to `<run_dir>/grading/` is then given to the invoking user.
"""

import json
import logging
import os
import uuid
from pathlib import Path

import docker
import requests
from docker.types import Mount

from environment.utils import parse_container_config
from experiments.make_submission import submission_line
from timeseriesgym.registry import Competition
from timeseriesgym.utils import get_timestamp

GRADER_IMAGE = "timeseriesgym-grader"
GRADING_TIMEOUT = 2 * 60 * 60  # seconds
GRADING_DIR = "grading"  # in the run directory
DATA_DIR = "/data"  # the grader's data directory, in its container
# taken from the agent's container configuration; GPUs, runtime and privileges are not
RESOURCE_KEYS = ("mem_limit", "memswap_limit", "nano_cpus", "cpuset_cpus", "shm_size")
# `timeseriesgym grade "$@"` as root, then the outputs to the invoking user, whatever its result
ENTRYPOINT = (
    'timeseriesgym "$@"; status=$?; '
    'chown -R -- "$TIMESERIESGYM_OUTPUT_OWNER" "$TIMESERIESGYM_OUTPUT_DIR" || status=1; '
    "exit $status"
)


def grader_mounts(competition: Competition, run_dir: Path) -> list[Mount]:
    """The competition's prepared data at the grader's data directory, and the run directory at
    its own path (so that the JSONL line and the report hold host paths), read-only except
    for its grading directory. Mounts, not `volumes`: those cannot take paths with a colon."""
    prepared = f"{DATA_DIR}/{competition.id}/prepared"
    run_dir, output_dir = run_dir.resolve().as_posix(), (run_dir.resolve() / GRADING_DIR).as_posix()
    private, public = competition.private_dir.resolve(), competition.public_dir.resolve()
    return [
        Mount(f"{prepared}/private", private.as_posix(), type="bind", read_only=True),
        Mount(f"{prepared}/public", public.as_posix(), type="bind", read_only=True),
        Mount(run_dir, run_dir, type="bind", read_only=True),
        Mount(output_dir, output_dir, type="bind", read_only=False),
    ]


def grade_run(
    client: docker.DockerClient,
    competition: Competition,
    run_dir: Path,
    image: str,
    container_config: dict,
    logger: logging.Logger,
    timeout: float = GRADING_TIMEOUT,
) -> dict:
    """Grades the outputs extracted into `run_dir`. Returns the run's competition report, with
    `report_path`, or `{"error": ...}` if grading failed (it never raises)."""
    try:
        return _grade_run(client, competition, run_dir, image, container_config, logger, timeout)
    except Exception as e:
        logger.error(f"Grading {run_dir} failed: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}


def _grade_run(client, competition, run_dir, image, container_config, logger, timeout) -> dict:
    run_dir = run_dir.resolve()
    output_dir = run_dir / GRADING_DIR
    output_dir.mkdir(exist_ok=True)
    submission = output_dir / "submission.jsonl"
    submission.write_text(json.dumps(submission_line(run_dir, competition)) + "\n")

    resources = {
        key: value
        for key, value in parse_container_config(container_config).items()
        if key in RESOURCE_KEYS and value is not None
    }
    owner = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "0:0"
    container = client.containers.create(
        image=image,
        entrypoint=["sh", "-c", ENTRYPOINT, "timeseriesgym-grade"],
        command=[
            "grade",
            "--submission",
            submission.as_posix(),
            "--output-dir",
            output_dir.as_posix(),
            "--data-dir",
            DATA_DIR,
        ],
        name=f"grader-{competition.id}-{get_timestamp()}-{uuid.uuid4().hex}",
        detach=True,
        network_mode="none",
        user="0:0",
        working_dir="/tmp",
        environment={
            "HOME": "/tmp",
            "TIMESERIESGYM_OUTPUT_OWNER": owner,
            "TIMESERIESGYM_OUTPUT_DIR": output_dir.as_posix(),
        },
        labels={"timeseriesgym.role": "grader"},
        mounts=grader_mounts(competition, run_dir),
        **resources,
    )
    logger.info(f"Grading in container {container.name}")
    try:
        container.start()
        try:
            exit_code = container.wait(timeout=timeout)["StatusCode"]
        except requests.exceptions.RequestException:  # the wait timed out
            container.kill()
            raise TimeoutError(f"grading took more than {timeout} seconds") from None
        logs = container.logs().decode("utf-8", errors="replace")
    finally:
        container.remove(force=True)

    (output_dir / "grader.log").write_text(logs)
    for line in logs.splitlines():
        logger.info(f"[Grader] {line}")
    if exit_code != 0:
        raise RuntimeError(f"the grader exited with code {exit_code} (see {output_dir}/grader.log)")
    reports = sorted(output_dir.glob("*_grading_report.json"))
    if not reports:
        raise RuntimeError(f"the grader wrote no report to {output_dir}")
    (report,) = json.loads(reports[-1].read_text())["competition_reports"]
    return {"report_path": reports[-1].as_posix(), **report}
