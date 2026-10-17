"""The agent's format check, `validate_submission` (environment/validate_submission.py): the
contract it checks submissions against, and its installation in the agent's container.

The contract describes the challenge's public sample submission (and, for folder challenges, the
files their graders read). It holds nothing from the private data, so the agent may read it.
"""

import json

import docker
from docker.models.containers import Container

from environment.utils import put_file
from environment.validate_submission import CONTRACT_PATH, describe_sample
from timeseriesgym.registry import Competition
from timeseriesgym.utils import get_module_dir, load_yaml

CONTRACT_VERSION = 1
SAMPLE_SUFFIXES = (".csv", ".npy", ".h5", ".hdf5", ".pkl")
# grade_feature_enhancement reads the predictions from this file of the submission folder
FEATURE_ENHANCEMENT_PREDICTIONS = "submission.csv"


class OutdatedImageError(RuntimeError):
    """The agent's image was built from a timeseriesgym-env without the format check."""


def _submission_format(competition: Competition) -> dict:
    """The optional `submission_format` options of the competition's config.yaml (see
    documentation/adding_new_challenges.md), e.g. `fixed_rows: false` for detected events."""
    config = load_yaml(get_module_dir() / "competitions" / competition.id / "config.yaml")
    return config.get("submission_format") or {}


def _describe(competition: Competition, options: dict) -> dict:
    sample = competition.sample_submission
    return {
        **describe_sample(sample, **options),
        "sample": sample.relative_to(competition.public_dir).as_posix(),
    }


def _code_contract(coding_config: dict) -> dict:
    """The classes, methods and functions the code grader runs, with their arguments."""

    def signature(spec: dict | None) -> dict:
        spec = spec or {}
        return {"args": list(spec.get("required_args") or {}), "exact": spec.get("exact_match")}

    return {
        "format": "py",
        "classes": {
            name: {m: signature(s) for m, s in (spec.get("required_methods") or {}).items()}
            for name, spec in (coding_config.get("required_classes") or {}).items()
        },
        "functions": {
            name: signature(spec)
            for name, spec in (coding_config.get("required_functions") or {}).items()
        },
    }


def submission_contract(
    competition: Competition, data_dir: str, submission_dir: str, code_dir: str
) -> dict:
    """The contract of `competition`'s submissions, for an agent container with the public
    data at `data_dir` and the given submission and code folders."""
    contract = {
        "version": CONTRACT_VERSION,
        "competition_id": competition.id,
        "data_dir": data_dir,
        "submission_dir": submission_dir,
        "code_dir": code_dir,
    }
    options = _submission_format(competition)
    sample = competition.sample_submission
    hps = competition.hyperparameter_search_config
    fe = competition.feature_enhancement_config
    if hps is not None:
        predictions = _describe(competition, options)
        model = {"format": "any"}  # loaded only when the run is graded
        contract["kind"] = "folder"
        contract["files"] = {
            hps["improved_submission_name"]: predictions,
            hps["baseline_submission_name"]: predictions,
            hps["improved_model_name"]: model,
            hps["baseline_model_name"]: model,
        }
        contract["solution_code"] = True  # the graders check the code for a search
    elif fe is not None:
        contract["kind"] = "folder"
        contract["files"] = {FEATURE_ENHANCEMENT_PREDICTIONS: _describe(competition, options)}
        contract["project_files"] = [fe["model_code_file"], fe["training_file"]]
        contract["optional_project_files"] = [f"conf/{fe['config_file']}"]
    elif competition.predictor_config is not None:
        from timeseriesgym.predictor.grading import settings

        config = settings(competition)
        contract["kind"] = "predictor"
        contract["predictor"] = {
            "dir": "predictor",  # in the submission folder: what make_submission.py points at
            "tools": f"{data_dir}/predictor_tools",  # the grader's format check, contract.py
            "max_bytes": int(config["max_bytes"]),
            "max_files": int(config["max_files"]),
        }
    elif competition.coding_config is not None:
        file_contract = _code_contract(competition.coding_config)
        file_contract["suffix"] = sample.suffix
        file_contract["sample"] = sample.relative_to(competition.public_dir).as_posix()
        contract["kind"] = "file"
        contract["file"] = file_contract
    elif sample.is_file() and sample.suffix in SAMPLE_SUFFIXES:
        contract["kind"] = "file"
        contract["file"] = _describe(competition, options)
    else:
        contract["kind"] = "none"
        contract["note"] = (
            f"the sample submission of {competition.id} is not a file of a type it can check"
        )
    return contract


def install_contract(container: Container, contract: dict) -> None:
    """Writes the contract to CONTRACT_PATH in the (created, not yet started) container."""
    data = json.dumps(contract, indent=1).encode()
    try:
        ok = put_file(container, CONTRACT_PATH.as_posix(), data)
    except docker.errors.NotFound:
        ok = False
    if not ok:
        raise OutdatedImageError(
            f"The agent's image has no {CONTRACT_PATH.parent}: it was built from a "
            "timeseriesgym-env that predates the format check, whose grading server mounted the "
            "test answers in the agent's container. Rebuild timeseriesgym-env and then the "
            "agent's image (agents/README.md, 'Migrating from the Grading Server')."
        )
