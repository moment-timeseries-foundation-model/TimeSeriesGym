"""Compile a run group's submissions into the JSONL file that `timeseriesgym grade` reads.

    python experiments/make_submission.py --metadata runs/<run-group>/metadata.json \
        --output runs/<run-group>/submission.jsonl

Each successful run in metadata.json contributes one line
{"competition_id": ..., "submission_path": ...}. The path is the submission file whose suffix
matches the challenge's sample submission, or the submission directory for hyperparameter-search
and feature-enhancement challenges. Hyperparameter-search lines also carry "solution_file_path":
the run's code directory (the agent's /home/code), which the graders require and search for a
hyperparameter search. Runs without a usable submission are skipped with a warning.
"""

import argparse
import json
from pathlib import Path

from timeseriesgym.registry import registry
from timeseriesgym.utils import get_logger

logger = get_logger(__name__)


def submission_path_for(run_dir: Path, competition) -> Path | None:
    submission_dir = run_dir / "submission"
    if not submission_dir.is_dir():
        return None
    if (
        competition.hyperparameter_search_config is not None
        or competition.feature_enhancement_config is not None
    ):
        return submission_dir  # their graders read several files from the folder
    suffix = competition.sample_submission.suffix
    candidates = sorted(p for p in submission_dir.iterdir() if p.is_file() and p.suffix == suffix)
    if len(candidates) != 1:
        logger.warning(f"{run_dir}: expected one `*{suffix}` submission, found {len(candidates)}")
        return None
    return candidates[0]


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
        if not run.get("success"):
            logger.warning(f"{run_id}: run failed, skipped")
            continue
        if "competition_id" not in run or "run_dir" not in run:
            logger.warning(f"{run_id}: metadata from an older run_agent.py, skipped")
            continue
        competition = reg.get_competition(run["competition_id"])
        path = submission_path_for(Path(run["run_dir"]), competition)
        if path is None:
            continue
        line = {"competition_id": competition.id, "submission_path": str(path)}
        if competition.hyperparameter_search_config is not None:
            # passed even if missing: the graders then reject the run (no solution code)
            line["solution_file_path"] = str(Path(run["run_dir"]) / "code")
        lines.append(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(line) + "\n" for line in lines))
    logger.info(f"Wrote {len(lines)} submissions to {args.output}")


if __name__ == "__main__":
    main()
