"""Compile a run group's submissions into the JSONL file that `timeseriesgym grade` reads.

    python experiments/make_submission.py --metadata runs/<run-group>/metadata.json \
        --output runs/<run-group>/submission.jsonl

Every run in metadata.json contributes one line {"competition_id": ..., "submission_path": ...},
failed runs included, so that the grading report counts every run: a run without a usable
submission points at a path that does not exist, which is graded as invalid. The path is the
submission file whose suffix matches the challenge's sample submission (the most recently
written one if there are several: the agent instructions say only the last file is evaluated),
or the submission directory for hyperparameter-search and feature-enhancement challenges, or
the predictor directory (`submission/predictor`) for the predictor challenges.
Hyperparameter-search lines also carry "solution_file_path": the run's code directory (the
agent's /home/code), which the graders require and search for a hyperparameter search.
Feature-enhancement lines carry "solution_files_path": the folder with the run's model code,
which is the submission directory, as the description asks, or else the code directory.
"""

import argparse
import json
from pathlib import Path

from timeseriesgym.registry import registry
from timeseriesgym.utils import get_logger

logger = get_logger(__name__)


def submission_path_for(run_dir: Path, competition) -> Path:
    submission_dir = run_dir / "submission"
    if (
        competition.hyperparameter_search_config is not None
        or competition.feature_enhancement_config is not None
    ):
        return submission_dir  # their graders read several files from the folder
    if competition.predictor_config is not None:
        return submission_dir / "predictor"  # the grader runs predict.py and its files
    suffix = competition.sample_submission.suffix
    candidates = []
    if submission_dir.is_dir():
        candidates = [p for p in submission_dir.iterdir() if p.is_file() and p.suffix == suffix]
    if not candidates:
        logger.warning(f"{run_dir}: no `*{suffix}` submission, graded as missing")
        return submission_dir / f"submission{suffix}"
    last = max(candidates, key=lambda p: (p.stat().st_mtime, p.name))
    if len(candidates) > 1:
        logger.warning(f"{run_dir}: {len(candidates)} `*{suffix}` submissions, using the last")
    return last


def solution_files_path_for(run_dir: Path, competition) -> Path:
    """The feature-enhancement solution files: the description asks for them in the submission
    directory, and agent scaffolds keep their code in /home/code."""
    submission_dir = run_dir / "submission"
    if (submission_dir / competition.feature_enhancement_config["model_code_file"]).is_file():
        return submission_dir
    return run_dir / "code"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", default=None, help="TimeSeriesGym data directory")
    args = parser.parse_args()

    reg = registry.set_data_dir(Path(args.data_dir)) if args.data_dir else registry
    metadata = json.loads(args.metadata.read_text())
    lines = []
    for run_id, run in metadata["runs"].items():
        if "competition_id" not in run or "run_dir" not in run:
            logger.warning(f"{run_id}: metadata from an older run_agent.py, skipped")
            continue
        if not run.get("success"):  # e.g. the agent timed out after saving its submission
            exit_code = run.get("agent_exit_code")
            logger.warning(f"{run_id}: run failed (agent exit code {exit_code}), still graded")
        competition = reg.get_competition(run["competition_id"])
        run_dir = Path(run["run_dir"])
        path = submission_path_for(run_dir, competition)
        line = {"competition_id": competition.id, "submission_path": str(path)}
        if competition.hyperparameter_search_config is not None:
            # passed even if missing: the graders then reject the run (no solution code)
            line["solution_file_path"] = str(run_dir / "code")
        if competition.feature_enhancement_config is not None:
            line["solution_files_path"] = str(solution_files_path_for(run_dir, competition))
        lines.append(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(line) + "\n" for line in lines))
    logger.info(f"Wrote {len(lines)} submissions to {args.output}")


if __name__ == "__main__":
    main()
