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

logger = get_logger(__name__)

SCORE_KEYS = (
    "defined_classes",
    "initialized_classes",
    "defined_class_methods",
    "executed_class_methods",
    "defined_functions",
    "executed_functions",
)


def run_submission(work: Path, job: dict) -> None:
    """The sandbox half of grade_code: the native grader up to where it reads the answers."""
    coding_config = sandbox.job_coding_config(work, job)

    # load the submission module
    submission_module = sandbox.load_module(work / job["submission"])

    # load the grading datasets
    load_grading_datasets(coding_config)

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

    # task-specific grading: model test accuracy
    model = initialize_class(
        submission_module, "ResNet", coding_config["required_classes"]["ResNet"]
    )
    train_func = load_attr(submission_module, attr_name="train", attr_type="function")
    predict_func = load_attr(submission_module, attr_name="predict", attr_type="function")
    predicted, two_outputs, error = False, False, None

    if model and train_func and predict_func:
        predicted = True
        try:
            train_args = coding_config["required_functions"]["train"]["required_args"]
            train_args["model"] = model
            model = train_func(**train_args)

            predict_args = coding_config["required_functions"]["predict"]["required_args"]
            predict_args["model"] = model
            outputs = predict_func(**predict_args)
        except Exception as e:
            logger.error(f"Error training and predicting for ResNet: {e}")
            outputs = None

        if isinstance(outputs, tuple) and len(outputs) == 2:
            two_outputs = True
            try:
                # outputs[0] is the dummy test labels, true labels are in answers
                # outputs[1] is the model predictions in (n_samples, n_classes)
                sandbox.save_array(work, "preds.npy", np.argmax(outputs[1], axis=1))
            except Exception as e:
                error = f"{type(e).__name__}: {e}"[:2000]

    result = {"predicted": predicted, "two_outputs": two_outputs, "error": error}
    sandbox.write_json(work, "result.json", {"scores": scores, **result})


def grade_code(
    submission_file_path: Path,
    answers: pd.DataFrame,
    coding_config: dict,
    private_paths: tuple[Path, ...] | None = None,
) -> dict:
    # The submission runs in the grading sandbox (run_submission); this compares its
    # predictions with the answers.
    with sandbox.run_code(
        f"{__name__}:run_submission", submission_file_path, coding_config, private_paths
    ) as run:
        result = run.result()
        scores = sandbox.scores(result, SCORE_KEYS)
        if not all(isinstance(result.get(key), bool) for key in ("predicted", "two_outputs")):
            raise InvalidSubmissionError("Submission invalid! Unusable results from the sandbox.")
        preds = run.array("preds.npy")
    test_accuracy = 0.0

    if result["predicted"]:
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

    # the synthetic time series dataset is an easy task
    # the correct ResNet implementation should achieve > 80% accuracy with the provided parameters
    if test_accuracy > coding_config["required_functions"]["predict"]["expected_accuray"]:
        scores["test_metric"] = "1/1"
    else:
        scores["test_metric"] = "0/1"

    return scores
