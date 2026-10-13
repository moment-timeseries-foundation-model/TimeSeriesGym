import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from timeseriesgym.grade_code_helper import (
    grade_function,
    load_attr,
    load_grading_datasets,
)
from timeseriesgym.utils import get_logger

from . import hidden_variant

logger = get_logger(__name__)


def _case_passes(stomp_func, args: dict, ref_data, query_data, expected, case: str) -> bool:
    """Whether `stomp` on these inputs returns the `expected` columns (mp, pi, ...) and, for
    join similarity, None for the right and left profiles."""
    try:
        query = None if query_data is None else np.array(query_data)
        outputs = stomp_func(**{**args, "ref_data": np.array(ref_data), "query_data": query})
    except Exception as e:
        logger.error(f"Error calling the stomp function ({case}): {e}")
        return False
    if not (isinstance(outputs, tuple) and len(outputs) == 6):
        return False
    try:
        mp_outputs = np.column_stack(outputs[: expected.shape[1]])
        is_output_close = mp_outputs.shape == expected.shape and np.allclose(
            mp_outputs, expected, rtol=1e-3, atol=1e-5
        )
    except Exception as e:
        logger.error(f"Error testing stomp outputs for {case}: {e}")
        return False
    # rmp, rpi, lmp, lpi should all be None in join similarity
    return bool(is_output_close) and (query_data is None or all(o is None for o in outputs[2:]))


def grade_code(
    submission_file_path: Path,
    answers: pd.DataFrame,
    coding_config: dict,
) -> dict:
    # load the submission module
    module_name = submission_file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, submission_file_path)
    submission_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submission_module)

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

    # task-specific grading: outputs need to be consistent with the answers. Each test case
    # passes only if the outputs also match on its hidden variant (hidden_variant.py).
    stomp_func = load_attr(submission_module, attr_name="stomp", attr_type="function")
    test_metric_pass, total_test_metric = 0, 2

    if stomp_func:
        stomp_args = coding_config["required_functions"]["stomp"]["required_args"]
        ref_data = stomp_args["ref_data"]
        query_data = np.load(coding_config["input_data_dir"] / "query_data.npy")
        true_outputs = answers.to_numpy()
        n_query_windows = len(query_data) - int(stomp_args["window_size"]) + 1
        variant = hidden_variant.transform

        # test case in config.yaml: only ref_data is provided (self-join)
        self_join = true_outputs[:, :6]
        cases = [
            (ref_data, None, self_join, "self-join"),
            (variant(ref_data), None, hidden_variant.expected_self_join(self_join), "self-join"),
        ]
        if all(_case_passes(stomp_func, stomp_args, *case) for case in cases):
            test_metric_pass += 1

        # task-specific test case: both ref_data and query_data are provided (join similarity)
        join = true_outputs[:, 6:]
        variant_join = hidden_variant.expected_join(join, n_query_windows)
        cases = [
            (ref_data, query_data, join, "join similarity"),
            (variant(ref_data), variant(query_data), variant_join, "join similarity"),
        ]
        if all(_case_passes(stomp_func, stomp_args, *case) for case in cases):
            test_metric_pass += 1

    scores["test_metric"] = f"{test_metric_pass}/{total_test_metric}"
    return scores
