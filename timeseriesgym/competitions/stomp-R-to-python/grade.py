from pathlib import Path

import numpy as np
import pandas as pd

from timeseriesgym import sandbox
from timeseriesgym.grade_code_helper import (
    grade_function,
    load_attr,
    load_grading_datasets,
)
from timeseriesgym.utils import InvalidSubmissionError, get_logger

from . import hidden_variant

logger = get_logger(__name__)

SCORE_KEYS = (
    "defined_classes",
    "initialized_classes",
    "defined_class_methods",
    "executed_class_methods",
    "defined_functions",
    "executed_functions",
)
# the calls of `stomp`: each test case on the public inputs and on its hidden variant
# (hidden_variant.py), whose inputs the sandbox computes from the public ones
CASES = ("self", "self_variant", "join", "join_variant")


def _keep_outputs(work: Path, case: str, outputs, n: int) -> dict:
    """What the comparison with the answers needs of one call's outputs, saved right away (a
    later call could change arrays it returned): whether they are 6 values, the first `n`
    stacked as columns (out/<case>.npy), and whether the other ones are all None."""
    if not (isinstance(outputs, tuple) and len(outputs) == 6):
        return {"six": False}
    try:
        sandbox.save_array(work, f"{case}.npy", np.column_stack(outputs[:n]))
        return {"six": True, "rest_none": all(o is None for o in outputs[n:]), "error": None}
    except Exception as e:
        return {"six": True, "error": f"{type(e).__name__}: {e}"[:2000]}


def run_submission(work: Path, job: dict) -> None:
    """The sandbox half of grade_code: the native grader up to where it reads the answers."""
    coding_config = sandbox.job_coding_config(work, job)

    # load the submission module
    submission_module = sandbox.load_module(work / job["submission"])

    # load the grading datasets
    load_grading_datasets(coding_config)

    # grade function definitions and executions
    function_scores = grade_function(submission_module, coding_config)

    # combine scores
    scores = {
        "defined_classes": None,
        "initialized_classes": None,
        "defined_class_methods": None,
        "executed_class_methods": None,
        "defined_functions": f"{function_scores['defined_functions']}/"
        f"{function_scores['total_functions']}",
        "executed_functions": f"{function_scores['executed_functions']}/"
        f"{function_scores['total_functions']}",
    }

    # task-specific grading: the outputs of each test case and of its hidden variant, which
    # the trusted half compares with the answers
    stomp_func = load_attr(submission_module, attr_name="stomp", attr_type="function")
    cases = None

    if stomp_func:
        stomp_args = coding_config["required_functions"]["stomp"]["required_args"]
        ref_data = stomp_args["ref_data"]
        query_data = np.load(coding_config["input_data_dir"] / "query_data.npy")
        variant = hidden_variant.transform
        inputs = {
            # test case in config.yaml: only ref_data is provided (self-join)
            "self": (ref_data, None),
            "self_variant": (variant(ref_data), None),
            # task-specific test case: both ref_data and query_data are provided (join similarity)
            "join": (ref_data, query_data),
            "join_variant": (variant(ref_data), variant(query_data)),
        }
        cases = {}
        for case, (ref, query) in inputs.items():
            label = "self-join" if query is None else "join similarity"
            try:
                query = None if query is None else np.array(query)
                outputs = stomp_func(
                    **{**stomp_args, "ref_data": np.array(ref), "query_data": query}
                )
            except Exception as e:
                logger.error(f"Error calling the stomp function ({label}): {e}")
                outputs = None
            cases[case] = _keep_outputs(work, case, outputs, 6 if query is None else 2)

    sandbox.write_json(work, "result.json", {"scores": scores, "cases": cases})


def _case(cases, case: str) -> dict:
    info = cases.get(case) if isinstance(cases, dict) else None
    if not (isinstance(info, dict) and isinstance(info.get("six"), bool)):
        raise InvalidSubmissionError("Submission invalid! Unusable results from the sandbox.")
    return info


def _case_passes(info: dict, outputs, expected: np.ndarray, join: bool, case: str) -> bool:
    """Whether one call returned the `expected` columns (mp, pi, ...) and, for join
    similarity, None for the right and left profiles."""
    if not info["six"]:
        return False
    try:
        if outputs is None:
            raise ValueError(info.get("error"))
        is_output_close = outputs.shape == expected.shape and np.allclose(
            outputs, expected, rtol=1e-3, atol=1e-5
        )
    except Exception as e:
        logger.error(f"Error testing stomp outputs for {case}: {e}")
        return False
    # rmp, rpi, lmp, lpi should all be None in join similarity
    return bool(is_output_close) and (not join or info.get("rest_none") is True)


def grade_code(
    submission_file_path: Path,
    answers: pd.DataFrame,
    coding_config: dict,
    private_paths: tuple[Path, ...] | None = None,
) -> dict:
    # The submission runs in the grading sandbox (run_submission); this compares its outputs
    # with the answers. Each test case passes only if the outputs also match on its hidden
    # variant, whose expected outputs are derived from the answers (hidden_variant.py).
    with sandbox.run_code(
        f"{__name__}:run_submission", submission_file_path, coding_config, private_paths
    ) as run:
        result = run.result()
        scores = sandbox.scores(result, SCORE_KEYS)
        cases = result.get("cases")
        if cases is not None:
            cases = {case: _case(cases, case) for case in CASES}
            arrays = {case: run.array(f"{case}.npy") for case in cases}

    test_metric_pass, total_test_metric = 0, 2

    if cases is not None:
        stomp_args = coding_config["required_functions"]["stomp"]["required_args"]
        query_data = np.load(coding_config["input_data_dir"] / "query_data.npy")
        n_query_windows = len(query_data) - int(stomp_args["window_size"]) + 1
        true_outputs = answers.to_numpy()
        self_join, join = true_outputs[:, :6], true_outputs[:, 6:]
        expected = {
            "self": self_join,
            "self_variant": hidden_variant.expected_self_join(self_join),
            "join": join,
            "join_variant": hidden_variant.expected_join(join, n_query_windows),
        }
        for test, label in (("self", "self-join"), ("join", "join similarity")):
            if all(
                _case_passes(cases[case], arrays[case], expected[case], test == "join", label)
                for case in (test, f"{test}_variant")
            ):
                test_metric_pass += 1

    scores["test_metric"] = f"{test_metric_pass}/{total_test_metric}"
    return scores
