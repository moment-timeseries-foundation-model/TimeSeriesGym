import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from timeseriesgym import sandbox
from timeseriesgym.grade_code_helper import (
    grade_class,
    grade_class_method,
    grade_function,
    initialize_class,
    load_attr,
    load_grading_datasets,
)
from timeseriesgym.utils import InvalidSubmissionError, get_logger

from .training_budget import TrainingSampleCounter

logger = get_logger(__name__)

SCORE_KEYS = (
    "defined_classes",
    "initialized_classes",
    "defined_class_methods",
    "executed_class_methods",
    "defined_functions",
    "executed_functions",
)


def _test_outputs(submission_module, coding_config: dict):
    """Build, train and run a fresh ResNet with the grading arguments: the outputs of
    `predict`, or None. The argument dicts in `coding_config` are not changed."""
    model = initialize_class(
        submission_module, "ResNet", coding_config["required_classes"]["ResNet"]
    )
    train_func = load_attr(submission_module, attr_name="train", attr_type="function")
    predict_func = load_attr(submission_module, attr_name="predict", attr_type="function")
    if not (model and train_func and predict_func):
        return None
    try:
        train_args = coding_config["required_functions"]["train"]["required_args"]
        model = train_func(**{**train_args, "model": model})

        predict_args = coding_config["required_functions"]["predict"]["required_args"]
        return predict_func(**{**predict_args, "model": model})
    except Exception as e:
        logger.error(f"Error training and predicting for ResNet: {e}")
        return None


def run_submission(work: Path, job: dict) -> None:
    """The sandbox half of grade_code: the native grader up to where it reads the answers."""
    coding_config = sandbox.job_coding_config(work, job)

    # load the submission module
    path = work / job["submission"]
    spec = importlib.util.spec_from_file_location(path.stem, path)
    submission_module = importlib.util.module_from_spec(spec)

    # The test model may be trained on at most n_epochs passes over train_data. The count
    # starts before the submission is imported and the test runs before any other call of
    # `train`, so no training outside it (at import, or kept from another call) goes uncounted.
    counter = TrainingSampleCounter(lambda: getattr(submission_module, "ResNet", None))
    with counter:
        spec.loader.exec_module(submission_module)

        # load the grading datasets
        load_grading_datasets(coding_config)
        train_args = coding_config["required_functions"]["train"]["required_args"]
        sample_budget = int(train_args["n_epochs"]) * len(train_args["train_data"])

        # task-specific grading: model test accuracy
        outputs = _test_outputs(submission_module, coding_config)

    # the test predictions, saved before any other call of the submission can change them
    two_outputs, error = isinstance(outputs, tuple) and len(outputs) == 2, None
    if two_outputs:
        try:
            # outputs[0] is the dummy test labels, true labels are in answers
            # outputs[1] is the model predictions in (n_samples, n_classes)
            sandbox.save_array(work, "preds.npy", np.argmax(outputs[1], axis=1))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"[:2000]

    # grade class definitions and initializations
    class_scores = grade_class(submission_module, coding_config)

    # grade class methods definitions and executions
    class_method_scores = grade_class_method(submission_module, coding_config)

    # task-specific preparation: need to initialize the ResNet class
    # to grade the train and predict functions
    model = initialize_class(
        submission_module, "ResNet", coding_config["required_classes"]["ResNet"]
    )
    if model is not None:
        coding_config["required_functions"]["train"]["required_args"]["model"] = model
        coding_config["required_functions"]["predict"]["required_args"]["model"] = model

    # grade function definitions and executions
    function_scores = grade_function(submission_module, coding_config)

    # combine all scores
    scores = {
        "defined_classes": f"{class_scores['defined_classes']}/{class_scores['total_classes']}",
        "initialized_classes": f"{class_scores['initialized_classes']}/"
        f"{class_scores['total_classes']}",
        "defined_class_methods": f"{class_method_scores['defined_class_methods']}/"
        f"{class_method_scores['total_class_methods']}",
        "executed_class_methods": f"{class_method_scores['executed_class_methods']}/"
        f"{class_method_scores['total_class_methods']}",
        "defined_functions": f"{function_scores['defined_functions']}/"
        f"{function_scores['total_functions']}",
        "executed_functions": f"{function_scores['executed_functions']}/"
        f"{function_scores['total_functions']}",
    }

    result = {
        "two_outputs": two_outputs,
        "error": error,
        "samples": counter.samples,
        "sample_budget": sample_budget,
    }
    sandbox.write_json(work, "result.json", {"scores": scores, **result})


def _count(result: dict, key: str) -> int:
    value = result.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidSubmissionError("Submission invalid! Unusable results from the sandbox.")
    return value


def grade_code(
    submission_file_path: Path,
    answers: pd.DataFrame,
    coding_config: dict,
    private_paths: tuple[Path, ...] | None = None,
) -> dict:
    # The submission runs in the grading sandbox (run_submission), which also counts the
    # samples the test model is trained on; this compares its predictions with the answers.
    with sandbox.run_code(
        f"{__name__}:run_submission", submission_file_path, coding_config, private_paths
    ) as run:
        result = run.result()
        scores = sandbox.scores(result, SCORE_KEYS)
        if not isinstance(result.get("two_outputs"), bool):
            raise InvalidSubmissionError("Submission invalid! Unusable results from the sandbox.")
        samples, sample_budget = _count(result, "samples"), _count(result, "sample_budget")
        preds = run.array("preds.npy")

    test_accuracy = 0.0
    true_labels = answers["label"].to_numpy()
    if result["two_outputs"]:
        try:
            if preds is None:
                raise ValueError(result.get("error"))
            if len(preds) == len(true_labels):
                test_accuracy = np.mean(preds == true_labels)
        except Exception as e:
            logger.error(f"Error computing accuracy for ResNet predictions: {e}")
            test_accuracy = 0.0

    within_budget = samples <= sample_budget
    if not within_budget:
        logger.warning(
            f"The test model was trained on {samples} samples, more than "
            f"n_epochs * len(train_data) = {sample_budget}: the test fails"
        )

    # the synthetic time series dataset is an easy task
    # the correct ResNet implementation should achieve > 80% accuracy with the provided parameters
    expected_accuracy = coding_config["required_functions"]["predict"]["expected_accuray"]
    if within_budget and test_accuracy > expected_accuracy:
        scores["test_metric"] = "1/1"
    else:
        scores["test_metric"] = "0/1"

    return scores
