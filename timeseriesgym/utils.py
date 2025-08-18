import difflib
import importlib
import json
import logging
import os
import pickle
import sys
import time
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from logging import Logger
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import py7zr
import torch
import torch.nn as nn
import yaml
from pandas import DataFrame
from tqdm.auto import tqdm


class Model(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """Loads the model from the given path."""
        pass


class PickleModel(Model):
    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def load(cls, path: str) -> "PickleModel":
        with open(path, "rb") as f:
            raw_model = pickle.load(f)
        return cls(raw_model)


class TorchModel(Model):
    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def load(cls, path: str) -> nn.Module:
        raw_model = torch.jit.load(path)
        return cls(raw_model)


def purple(str: str) -> str:
    return f"\033[1;35m{str}\033[0m"


def authenticate_kaggle_api() -> "KaggleApi":  # noqa: F821
    """Authenticates the Kaggle API and returns an authenticated API object, or raises an error if
    authentication fails."""
    try:
        # only import when necessary; otherwise kaggle asks for API key on import
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competitions_list()  # a cheap op that requires authentication
        return api
    except Exception as e:
        logger.error(f"Authentication failed: {e!s}")
        raise PermissionError(
            "Kaggle authentication failed! Please ensure you have valid Kaggle API credentials "
            "configured. Refer to the Kaggle API documentation for guidance on setting up "
            "your API token."
        ) from e


def read_jsonl(file_path: str, skip_commented_out_lines: bool = False) -> list[dict]:
    """
    Read a JSONL file and return a list of dictionaries of its content.

    Args:
        file_path (str): Path to the JSONL file.
        skip_commented_out_lines (bool): If True, skip commented out lines.

    Returns:
        list[dict]: List of dictionaries parsed from the JSONL file.
    """
    result = []
    with open(file_path) as f:
        if skip_commented_out_lines:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                result.append(json.loads(line))
        else:
            return [json.loads(line) for line in f]
    return result


def load_arbitrary_h5(root):
    if isinstance(root, h5py.Dataset):
        return root[()]

    root_dict = {}
    stack = [(root, root_dict)]

    while stack:
        group, output_dict = stack.pop()

        for name, child in group.items():
            if isinstance(child, h5py.Dataset):
                output_dict[name] = child[()]
            else:
                subgroup_dict = {}
                output_dict[name] = subgroup_dict
                stack.append((child, subgroup_dict))

    return root_dict


def load_answers(path_to_answers: Path) -> Any:
    supported_formats = [".csv", ".jsonl", ".h5", ".hdf5", ".pkl", ".npy"]

    if path_to_answers.suffix == ".csv":
        return read_csv(path_to_answers)
    elif path_to_answers.suffix == ".jsonl":
        return read_jsonl(str(path_to_answers))
    elif path_to_answers.suffix == ".h5" or path_to_answers.suffix == ".hdf5":
        with h5py.File(path_to_answers, "r") as answers_file:
            return load_arbitrary_h5(answers_file["labels"])
    elif path_to_answers.suffix == ".pkl":
        with open(path_to_answers, "rb") as f:
            return pickle.load(f)
    elif path_to_answers.suffix == ".npy":
        return np.load(path_to_answers)
    else:
        raise InvalidSubmissionError(
            f"Unsupported file format for answers: {path_to_answers.suffix}. "
            f"Supported formats are {supported_formats}"
        )


def load_submissions(path_to_submission: Path) -> Any:
    supported_formats = [".csv", ".h5", ".hdf5", ".pkl", ".npy"]

    if not path_to_submission.is_file():
        return InvalidSubmissionError(
            f"Submission invalid! Submission file {path_to_submission} does not exist."
        )

    if path_to_submission.suffix.lower() == ".csv":
        return read_csv(path_to_submission)

    elif path_to_submission.suffix.lower() == ".h5" or path_to_submission.suffix.lower() == ".hdf5":
        with h5py.File(path_to_submission, "r") as submission_file:
            return load_arbitrary_h5(submission_file["submission"])

    elif path_to_submission.suffix.lower() == ".pkl":
        with open(path_to_submission, "rb") as f:
            return pickle.load(f)

    elif path_to_submission.suffix == ".npy":
        return np.load(path_to_submission)

    else:
        raise InvalidSubmissionError(
            f"Unsupported file format for submission: {path_to_submission.suffix}. "
            f"Supported formats are {supported_formats}"
        )


def load_scripts(path_to_scripts: Path) -> str:
    if not path_to_scripts.is_file():
        return InvalidSubmissionError(
            f"Submission invalid! script file {path_to_scripts} does not exist."
        )

    with open(path_to_scripts) as f:
        script_contents = f.read()

    return script_contents


def load_models(path_to_models: Path) -> Model:
    if not path_to_models.is_file():
        return InvalidSubmissionError(
            f"Submission invalid! model file {path_to_models} does not exist."
        )

    if path_to_models.suffix.lower() == ".pt":
        model = TorchModel.load(path_to_models)
    elif path_to_models.suffix.lower() == ".pkl":
        model = PickleModel.load(path_to_models)
    else:
        raise InvalidSubmissionError(
            f"Unsupported file format for model: {path_to_models.suffix}. "
            f"Supported formats are .pt and .pkl"
        )
    return model


def get_runs_dir(run_dir) -> Path:
    """Returns an absolute path to the directory storing runs."""
    if run_dir:
        return Path(run_dir)
    return get_module_dir().parent / "runs"


def get_module_dir() -> Path:
    """Returns an absolute path to the TimeSeriesGym module."""

    path = Path(__file__).parent.resolve()

    assert (
        path.name == "timeseriesgym"
    ), f"Expected the module directory to be `timeseriesgym`, but got `{path.name}`."

    return path


def get_repo_dir() -> Path:
    """Returns an absolute path to the repository directory."""

    return get_module_dir().parent


def generate_run_id(competition_id: str, agent_id: str, run_group: str | None = None) -> str:
    """Creates a unique run ID for a specific competition and agent combo"""

    timestamp = get_timestamp()
    long_id = str(uuid.uuid4())
    short_id = long_id[:8]

    if run_group:  # the timestamp and agent are already included in the run group name
        return f"{competition_id}_{long_id}"

    return f"{timestamp}_{competition_id}_{agent_id}_{short_id}"


def create_run_dir(
    competition_id: str | None = None,
    agent_id: str | None = None,
    run_group: str | None = None,
    run_dir: str | None = None,
) -> Path:
    """Creates a directory for the run."""

    assert competition_id is None or isinstance(
        competition_id, str
    ), f"Expected a string or None, but got `{type(competition_id).__name__}`."

    assert agent_id is None or isinstance(
        agent_id, str
    ), f"Expected a string or None, but got `{type(agent_id).__name__}`."

    assert run_group is None or isinstance(
        run_group, str
    ), f"Expected a string or None, but got `{type(run_group).__name__}`."

    assert run_dir is None or isinstance(
        run_dir, str
    ), f"Expected a string or None, but got `{type(run_dir).__name__}`."

    run_id = str(uuid.uuid4())

    if competition_id and agent_id:
        run_id = generate_run_id(competition_id, agent_id, run_group)

    root_run_dir = get_runs_dir(run_dir)
    run_dir = root_run_dir / run_id

    if run_group:
        run_dir = root_run_dir / run_group / run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    assert isinstance(run_dir, Path), f"Expected a `Path`, but got `{type(run_dir)}`."
    assert run_dir.is_dir(), f"Expected a directory, but got `{run_dir}`."

    return run_dir


def is_compressed(fpath: Path) -> bool:
    """Checks if the file is compressed."""

    return fpath.suffix in [".zip", ".tar", ".gz", ".tgz", ".tar.gz", ".rar", ".7z"]


def compress(src: Path, compressed: Path, exist_ok: bool = False) -> None:
    """Compresses the contents of a source directory to a compressed file."""
    assert src.exists(), f"Source directory `{src}` does not exist."
    assert src.is_dir(), f"Expected a directory, but got `{src}`."
    if not exist_ok:
        assert not compressed.exists(), f"Compressed file `{compressed}` already exists."

    tqdm_desc = f"Compressing {src.name} to {compressed.name}"
    file_paths = [path for path in src.rglob("*") if path.is_file()]
    total_files = len(file_paths)

    def zip_compress(src: Path, compressed: Path):
        with zipfile.ZipFile(compressed, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tqdm(file_paths, desc=tqdm_desc, unit="file", total=total_files):
                zipf.write(file_path, arcname=file_path.relative_to(src))

    def sevenz_compress(src: Path, compressed: Path):
        with py7zr.SevenZipFile(compressed, "w") as archive:
            for file_path in tqdm(file_paths, desc=tqdm_desc, unit="file", total=total_files):
                archive.write(file_path, arcname=file_path.relative_to(src))

    # Determine the compression format from the destination file suffix
    if compressed.suffix == ".zip":
        zip_compress(src, compressed)
    elif compressed.suffix == ".7z":
        sevenz_compress(src, compressed)
    else:
        raise NotImplementedError(f"Unsupported compression format: `{compressed.suffix}`.")


def extract(
    compressed: Path, dst: Path, recursive: bool = False, already_extracted: set | None = None
) -> None:
    if already_extracted is None:
        already_extracted = set()
    """Extracts the contents of a compressed file to a destination directory."""

    # pre-conditions
    assert compressed.exists(), f"File `{compressed}` does not exist."
    assert compressed.is_file(), f"Path `{compressed}` is not a file."
    assert is_compressed(compressed), f"File `{compressed}` is not compressed."

    if compressed.suffix == ".7z":
        with py7zr.SevenZipFile(compressed, mode="r") as ref:
            ref.extractall(dst)
    elif compressed.suffix == ".zip":
        with zipfile.ZipFile(compressed, "r") as ref:
            ref.extractall(dst)
    else:
        raise NotImplementedError(f"Unsupported compression format: `{compressed.suffix}`.")

    already_extracted.add(compressed)
    if recursive:
        to_extract = {
            fpath for fpath in set(dst.iterdir()) - already_extracted if is_compressed(fpath)
        }
        already_extracted.update(to_extract)

        for fpath in to_extract:
            extract(fpath, fpath.parent, recursive=True, already_extracted=already_extracted)


def is_empty(dir: Path) -> bool:
    """Checks if the directory is empty."""

    # pre-conditions
    assert isinstance(dir, Path), f"Expected a `Path`, but got `{type(dir)}`."
    assert dir.is_dir(), f"Expected a directory, but got `{dir}`."

    # body
    return not any(dir.iterdir())


def get_logger(name: str, level: int = logging.INFO, filename: Path | None = None) -> Logger:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        filename=filename,
    )
    return logging.getLogger(name)


def load_yaml(fpath: Path) -> dict:
    """Loads a YAML file and returns its contents as a dictionary."""

    assert isinstance(fpath, Path), f"Expected a `Path`, but got `{type(fpath)}`."
    assert fpath.exists(), f"File `{fpath}` does not exist."
    assert fpath.is_file(), f"Expected a file, but got `{fpath}`."
    assert fpath.suffix == ".yaml", f"Expected a YAML file, but got `{fpath}`."

    with open(fpath) as file:
        contents = yaml.safe_load(file)

    return contents


def in_ci() -> bool:
    """Checks if the code is running in GitHub CI."""

    return os.environ.get("GITHUB_ACTIONS") == "true"


def import_fn(fn_import_string: str) -> Callable:
    """
    Imports a function from a module given a string in the format
    `potentially.nested.module_name:fn_name`.

    Basically equivalent to `from potentially.nested.module_name import fn_name`.
    """
    module_name, fn_name = fn_import_string.split(":")
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


def get_path_to_callable(callable: Callable) -> Path:
    """
    Retrieves the file path of the module where the given callable is defined.

    Args:
        callable (Callable): The callable for which the module path is required.

    Returns:
        Path: The relative path to the module file from the current working directory.

    Raises:
        AssertionError: If the module does not have a file path.
    """

    file_path = getattr(sys.modules[callable.__module__], "__file__", None)
    assert file_path is not None, f"Module `{callable.__module__}` does not have a file path."
    return Path(os.path.relpath(file_path, start=os.getcwd()))


def get_diff(d: dict, other_d: dict, fromfile: str = "d", tofile: str = "other_d") -> str:
    """Finds the differences between two nested dictionaries and returns a diff string."""

    d_str = yaml.dump(d, sort_keys=True)
    other_d_str = yaml.dump(other_d, sort_keys=True)

    diff = difflib.unified_diff(
        d_str.splitlines(),
        other_d_str.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )

    return "\n".join(diff)


def read_csv(*args, **kwargs) -> DataFrame:
    """Reads a CSV file and returns a DataFrame with custom default kwargs."""

    try:
        new_default_kwargs = {"float_precision": "round_trip"}
        new_kwargs = {**new_default_kwargs, **kwargs}
        return pd.read_csv(*args, **new_kwargs)
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file empty! {args[0]}")
        return pd.DataFrame()


def read_parquet(*args, **kwargs) -> DataFrame:
    """Reads a parquet file and returns a DataFrame with custom default kwargs."""

    try:
        return pd.read_parquet(*args, **kwargs)
    except pd.errors.EmptyDataError:
        logger.warning(f"Parquet file empty! {args[0]}")
        return pd.DataFrame()


def get_timestamp() -> str:
    """Returns the current timestamp in the format `YYYY-MM-DDTHH-MM-SS-Z`."""

    return time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())


def get_null_rank_info() -> dict:
    """Returns a dictionary with null values for rank information."""

    ranks = {
        "gold_medal": False,
        "silver_medal": False,
        "bronze_medal": False,
        "above_median": False,
        "gold_threshold": -1,
        "silver_threshold": -1,
        "bronze_threshold": -1,
        "median_threshold": -1,
    }

    return ranks


class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass


def have_different_hyperparameters_torch(m1: nn.Module, m2: nn.Module):
    def extract_architecture(model: nn.Module):
        """
        Returns an ordered dict mapping layer-names to a tuple
        (layer-type, dict_of_relevant_hyperparams, parameter_shape_dict)
        """
        arch = OrderedDict()
        for name, module in model.named_modules():
            # skip the top-level container
            if name == "":
                continue

            layer_type = type(module).__name__
            # pull out the usual hyperparams for this layer
            params = {}
            for attr in (
                "in_features",
                "out_features",  # Linear
                "in_channels",
                "out_channels",  # Conv2d
                "kernel_size",
                "stride",  # Conv2d
                "padding",
                "bias",  # Conv2d, Linear
            ):
                if hasattr(module, attr):
                    params[attr] = getattr(module, attr)

            # capture shapes of any weight/bias tensors
            shapes = {}
            for p_name, p in module.named_parameters(recurse=False):
                shapes[p_name] = tuple(p.shape)

            arch[name] = (layer_type, params, shapes)
        return arch

    a1 = extract_architecture(m1)
    a2 = extract_architecture(m2)

    # find layers present in one but not the other
    only_in_1 = set(a1) - set(a2)
    only_in_2 = set(a2) - set(a1)

    if not (not only_in_1 and not only_in_2):
        return True

    # for layers in common, compare layer-type, hyperparams, shapes
    diffs = {}

    for name in set(a1).intersection(a2):
        t1, p1, s1 = a1[name]
        t2, p2, s2 = a2[name]
        layer_diffs = {}
        if t1 != t2:
            layer_diffs["type"] = (t1, t2)
        if s1 != s2:
            layer_diffs["param_shapes"] = (s1, s2)
        if layer_diffs:
            diffs[name] = layer_diffs

    if diffs == {}:
        return False
    else:
        return True


def have_different_hyperparameters(baseline_model: Model, improved_model: Model) -> bool:
    """
    Check if two models have different hyperparameters.
    """
    if type(baseline_model) != type(improved_model):
        raise InvalidSubmissionError(
            f"Models are of different types: {type(baseline_model)} and {type(improved_model)}"
        )

    if isinstance(baseline_model, TorchModel) and isinstance(improved_model, TorchModel):
        return have_different_hyperparameters_torch(baseline_model.model, improved_model.model)

    if isinstance(baseline_model, PickleModel) and isinstance(improved_model, PickleModel):
        baseline_dict = baseline_model.model.__dict__
        improved_dict = improved_model.model.__dict__

        for key in improved_dict:
            if key not in baseline_dict or improved_dict[key] is None:
                continue
            if baseline_dict[key] != improved_dict[key]:
                return True

    return False


def have_hyperparameter_searching_attempt(solution_code: str, packages_to_look_for=None) -> bool:
    """
    Check if the solution code contains hyperparameter searching attempts.
    """
    # Check for common hyperparameter searching libraries
    # TODO: add regex
    if packages_to_look_for is None:
        hyperparameter_searching_libraries = [
            "optuna",
            "ray.tune",
            "wandb",
            "mlflow",
            "hyperopt",
            "sklearn.model_selection",
            "bayes_opt",
            "sigopt",
            "nevergrad",
            "ax",
            "pytorch_lightning.tuner",
        ]
    else:
        hyperparameter_searching_libraries = packages_to_look_for

    for library in hyperparameter_searching_libraries:
        import_str = f"import {library}"
        from_str = f"from {library}"
        if import_str in solution_code:
            # count if used twice, so not just imported
            if solution_code.count(import_str) > 1:
                return True
        elif from_str in solution_code:
            # get actual function name
            function_names = solution_code.split(from_str)[1].split(",")
            for function_name in function_names:
                if solution_code.count(function_name.strip()) > 1:
                    return True

    return False


logger = get_logger(__name__)
