import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from timeseriesgym.grade_code_helper import (
    grade_class,
    grade_class_method,
    grade_function,
    initialize_class,
    load_attr,
    load_grading_datasets,
)
from timeseriesgym.utils import InvalidSubmissionError, get_logger

from .architecture import keras_trainable_parameters, trainable_parameters_in_use
from .training_budget import TrainingSampleCounter

logger = get_logger(__name__)


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


def _architecture_matches(submission_module, coding_config: dict) -> bool:
    """Whether a fresh ResNet's forward pass uses exactly the Keras model's number of trainable
    parameters for the grading arguments (architecture.py)."""
    resnet_config = coding_config["required_classes"]["ResNet"]
    init_args = resnet_config["required_methods"]["__init__"]["required_args"]
    expected = keras_trainable_parameters(
        int(init_args["input_shape"][0]),
        int(init_args["n_feature_maps"]),
        int(init_args["n_classes"]),
    )
    model = initialize_class(submission_module, "ResNet", resnet_config)
    if model is None:
        return False
    try:
        x = resnet_config["required_methods"]["forward"]["required_args"]["x"]
        used = trainable_parameters_in_use(model, x)
    except Exception as e:
        logger.error(f"Error counting the trainable parameters of ResNet: {e}")
        return False
    if used != expected:
        logger.warning(
            f"ResNet's forward pass uses {used} trainable parameters; the Keras model of "
            f"resnet.py has {expected} with these arguments: the test fails"
        )
    return used == expected


def grade_code(
    submission_file_path: Path,
    answers: pd.DataFrame,
    coding_config: dict,
) -> dict:
    # load the submission module
    module_name = submission_file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, submission_file_path)
    submission_module = importlib.util.module_from_spec(spec)

    # The test model may be trained on at most n_epochs passes over train_data. The count
    # starts before the submission is imported and the test runs before any other call of
    # `train`, so no training outside it (at import, or kept from another call) goes uncounted.
    counter = TrainingSampleCounter()
    with counter:
        spec.loader.exec_module(submission_module)

        # load the grading datasets
        load_grading_datasets(coding_config)
        train_args = coding_config["required_functions"]["train"]["required_args"]
        sample_budget = int(train_args["n_epochs"]) * len(train_args["train_data"])

        # task-specific grading: model test accuracy
        outputs = _test_outputs(submission_module, coding_config)

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

    # task-specific grading: the architecture (after the test model, outside its sample count)
    architecture_matches = _architecture_matches(submission_module, coding_config)

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

    test_accuracy = 0.0
    true_labels = answers["label"].to_numpy()
    if isinstance(outputs, tuple) and len(outputs) == 2:
        try:
            # outputs[0] is the dummy test labels, true labels are in answers
            # outputs[1] is the model predictions in (n_samples, n_classes)
            preds = np.argmax(outputs[1], axis=1)
            if len(preds) == len(true_labels):
                test_accuracy = np.mean(preds == true_labels)
        except Exception as e:
            logger.error(f"Error computing accuracy for ResNet predictions: {e}")
            test_accuracy = 0.0

    # A model better than any constant class that was trained on no sample the counter saw
    # (stored weights, a hand-written rule, or training it cannot see) is not graded.
    best_constant = float(answers["label"].value_counts(normalize=True).max())
    if counter.samples == 0 and test_accuracy > best_constant:
        raise InvalidSubmissionError(
            f"the test model reached accuracy {test_accuracy:.4f}, above the best constant class "
            f"({best_constant:.4f}), but no training sample was back-propagated through a layer "
            "with parameters: train it with `train`, through the model's layers"
        )
    within_budget = counter.samples <= sample_budget
    if not within_budget:
        logger.warning(
            f"The test model was trained on {counter.samples} samples, more than "
            f"n_epochs * len(train_data) = {sample_budget}: the test fails"
        )

    # the synthetic time series dataset is an easy task
    # the correct ResNet implementation should achieve > 80% accuracy with the provided parameters
    expected_accuracy = coding_config["required_functions"]["predict"]["expected_accuray"]
    if within_budget and architecture_matches and test_accuracy > expected_accuracy:
        scores["test_metric"] = "1/1"
    else:
        scores["test_metric"] = "0/1"

    return scores
