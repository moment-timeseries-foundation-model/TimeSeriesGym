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

logger = get_logger(__name__)

SCORE_KEYS = (
    "defined_classes",
    "initialized_classes",
    "defined_class_methods",
    "executed_class_methods",
    "defined_functions",
    "executed_functions",
)


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

    # task-specific grading: outputs need to be consistent with the answers
    stomp_func = load_attr(submission_module, attr_name="stomp", attr_type="function")
    cases = None

    if stomp_func:
        # test case in config.yaml: only ref_data is provided (self-join)
        try:
            stomp_args = coding_config["required_functions"]["stomp"]["required_args"]
            outputs = stomp_func(**stomp_args)
        except Exception as e:
            logger.error(f"Error calling the stomp function (self-join): {e}")
            outputs = None
        cases = {"self": _keep_outputs(work, "self", outputs, 6)}

        # task-specific test case: both ref_data and query_data are provided (join similarity)
        try:
            stomp_args = coding_config["required_functions"]["stomp"]["required_args"]
            stomp_args["query_data"] = np.load(coding_config["input_data_dir"] / "query_data.npy")
            outputs = stomp_func(**stomp_args)
        except Exception as e:
            logger.error(f"Error calling the stomp function (join similarity): {e}")
            outputs = None
        cases["join"] = _keep_outputs(work, "join", outputs, 2)

    sandbox.write_json(work, "result.json", {"scores": scores, "cases": cases})


def _case(cases, case: str) -> dict:
    info = cases.get(case) if isinstance(cases, dict) else None
    if not (isinstance(info, dict) and isinstance(info.get("six"), bool)):
        raise InvalidSubmissionError("Submission invalid! Unusable results from the sandbox.")
    return info


def grade_code(
    submission_file_path: Path,
    answers: pd.DataFrame,
    coding_config: dict,
    private_paths: tuple[Path, ...] | None = None,
) -> dict:
    # The submission runs in the grading sandbox (run_submission); this compares its outputs
    # with the answers.
    with sandbox.run_code(
        f"{__name__}:run_submission", submission_file_path, coding_config, private_paths
    ) as run:
        result = run.result()
        scores = sandbox.scores(result, SCORE_KEYS)
        cases = result.get("cases")
        if cases is not None:
            cases = {case: _case(cases, case) for case in ("self", "join")}
            arrays = {case: run.array(f"{case}.npy") for case in cases}

    test_metric_pass, total_test_metric = 0, 2

    if cases is not None:
        # test case in config.yaml: only ref_data is provided (self-join)
        true_outputs = answers.to_numpy()
        true_outputs = true_outputs[:, :6]  # outputs for self-join
        if cases["self"]["six"]:
            try:
                if arrays["self"] is None:
                    raise ValueError(cases["self"].get("error"))
                mp_outputs = arrays["self"]
                is_output_close = np.allclose(mp_outputs, true_outputs, rtol=1e-3, atol=1e-5)

                if is_output_close:
                    test_metric_pass += 1
            except Exception as e:
                logger.error(f"Error testing stomp outputs for self-join: {e}")

        # task-specific test case: both ref_data and query_data are provided (join similarity)
        true_outputs = answers.to_numpy()
        true_outputs = true_outputs[:, 6:]  # outputs for join similarity
        if cases["join"]["six"]:
            try:
                if arrays["join"] is None:
                    raise ValueError(cases["join"].get("error"))
                mp_outputs = arrays["join"]
                is_output_close = np.allclose(mp_outputs, true_outputs, rtol=1e-3, atol=1e-5)
                # rmp, rpi, lmp, lpi should all be None in join similarity
                is_output_none = cases["join"].get("rest_none") is True

                if is_output_close and is_output_none:
                    test_metric_pass += 1
            except Exception as e:
                logger.error(f"Error testing stomp outputs for join similarity: {e}")

    scores["test_metric"] = f"{test_metric_pass}/{total_test_metric}"
    return scores
