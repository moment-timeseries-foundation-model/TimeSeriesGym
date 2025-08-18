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

logger = get_logger(__name__)


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

    # task-specific grading: outputs need to be consistent with the answers
    stomp_func = load_attr(submission_module, attr_name="stomp", attr_type="function")
    test_metric_pass, total_test_metric = 0, 2

    if stomp_func:
        # test case in config.yaml: only ref_data is provided (self-join)
        try:
            stomp_args = coding_config["required_functions"]["stomp"]["required_args"]
            outputs = stomp_func(**stomp_args)
        except Exception as e:
            logger.error(f"Error calling the stomp function (self-join): {e}")
            outputs = None

        true_outputs = answers.to_numpy()
        true_outputs = true_outputs[:, :6]  # outputs for self-join
        if isinstance(outputs, tuple) and len(outputs) == 6:
            try:
                mp_outputs = np.column_stack(outputs)
                is_output_close = np.allclose(mp_outputs, true_outputs, rtol=1e-3, atol=1e-5)

                if is_output_close:
                    test_metric_pass += 1
            except Exception as e:
                logger.error(f"Error testing stomp outputs for self-join: {e}")

        # task-specific test case: both ref_data and query_data are provided (join similarity)
        try:
            stomp_args = coding_config["required_functions"]["stomp"]["required_args"]
            stomp_args["query_data"] = np.load(coding_config["input_data_dir"] / "query_data.npy")
            outputs = stomp_func(**stomp_args)
        except Exception as e:
            logger.error(f"Error calling the stomp function (join similarity): {e}")
            outputs = None

        true_outputs = answers.to_numpy()
        true_outputs = true_outputs[:, 6:]  # outputs for join similarity
        if isinstance(outputs, tuple) and len(outputs) == 6:
            try:
                mp_outputs = np.column_stack(outputs[:2])
                is_output_close = np.allclose(mp_outputs, true_outputs, rtol=1e-3, atol=1e-5)
                # rmp, rpi, lmp, lpi should all be None in join similarity
                is_output_none = all(o is None for o in outputs[2:])

                if is_output_close and is_output_none:
                    test_metric_pass += 1
            except Exception as e:
                logger.error(f"Error testing stomp outputs for join similarity: {e}")

    scores["test_metric"] = f"{test_metric_pass}/{total_test_metric}"
    return scores
