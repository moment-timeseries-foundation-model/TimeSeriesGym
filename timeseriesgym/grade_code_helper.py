import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from timeseriesgym.utils import get_logger

logger = get_logger(__name__)


def load_dataset_for_function(func_config: dict, input_data_dir: Path) -> None:
    required_args = func_config.get("required_args", {})
    for arg_name, arg_value in required_args.items():
        if isinstance(arg_value, dict) and "file_name" in arg_value:
            file_path = input_data_dir / arg_value["file_name"]
            if file_path.exists():
                if file_path.suffix == ".npy":
                    data = np.load(file_path)
                if file_path.suffix == ".pt":
                    data = torch.load(file_path, weights_only=False)
                func_config["required_args"][arg_name] = data
            else:
                raise FileNotFoundError(f"File {file_path} not found.")


def load_grading_datasets(coding_config: dict) -> None:
    input_data_dir = Path(coding_config["input_data_dir"])

    required_classes = coding_config.get("required_classes", {})
    for _, class_config in required_classes.items():
        required_methods = class_config.get("required_methods", {})
        for _, method_config in required_methods.items():
            load_dataset_for_function(method_config, input_data_dir)

    required_functions = coding_config.get("required_functions", {})
    for _, func_config in required_functions.items():
        load_dataset_for_function(func_config, input_data_dir)


def load_attr(submission_module: ModuleType, attr_name: str, attr_type: str = "class") -> Any:
    has_attr = hasattr(submission_module, attr_name)
    if has_attr:
        attr_module = getattr(submission_module, attr_name)
        if attr_type == "class":
            inspect_func = inspect.isclass
        elif attr_type == "method":
            inspect_func = inspect.ismethod
        elif attr_type == "function":
            inspect_func = inspect.isfunction

        if inspect_func(attr_module):
            return attr_module
    return None


def grade_class_definition(
    submission_module: ModuleType, class_name: str, class_config: dict
) -> int:
    """
    A class is considered successfully defined if:
    - the class name exists in the submission and it is a valid Python class
    - the class has all the required methods
    """
    class_module = load_attr(submission_module, class_name, attr_type="class")
    if class_module is None:
        return 0

    score = 0
    try:
        actual_methods = inspect.getmembers(class_module, predicate=inspect.isfunction)
        # Filter methods to only include those defined in the class
        # and not inherited from parent classes
        actual_methods = [
            method[0]
            for method in actual_methods
            if method[1].__qualname__.startswith(class_name + ".")
        ]
    except Exception as e:
        logger.error(f"Error extracting methods from class {class_name}: {e}")
        return 0

    required_methods = list(class_config.get("required_methods", {}).keys())
    score = int(all(method in actual_methods for method in required_methods))
    return score


def initialize_class(
    submission_module: ModuleType,
    class_name: str,
    class_config: dict,
) -> Any:
    """
    A class is considered successfully initialized if:
    - an class object can be initialized with the provided arguments
    """
    class_module = load_attr(submission_module, class_name, attr_type="class")
    if class_module is None:
        return None

    required_methods = class_config.get("required_methods", {})
    if "__init__" not in required_methods:
        return None

    init_args = required_methods.get("__init__", {}).get("required_args", {})
    try:
        obj = class_module(**init_args)
    except Exception as e:
        logger.error(f"Error initializing class {class_name}: {e}")
        return None
    return obj


def grade_class(submission_module: ModuleType, coding_config: dict) -> dict:
    """
    Each class is graded based on:
    - whether the class is defined correctly
    - whether the class can be initialized with the required arguments
    """
    total_class_defined = 0
    total_class_initialized = 0
    total_class_num = 0

    required_classes = coding_config.get("required_classes", {})
    for class_name, class_config in required_classes.items():
        # Check if the class is defined correctly
        score = grade_class_definition(submission_module, class_name, class_config)
        total_class_defined += score

        class_obj = initialize_class(submission_module, class_name, class_config)
        if class_obj is not None:
            total_class_initialized += 1

        total_class_num += 1

    scores = {
        "defined_classes": total_class_defined,
        "initialized_classes": total_class_initialized,
        "total_classes": total_class_num,
    }
    return scores


def grade_function_definition(
    submission_module: ModuleType,
    func_name: str,
    func_config: dict,
    class_method: bool = False,
) -> int:
    """
    A function is considered successfully defined if:
    - the function name exists in the submission and it is a valid Python function
    - the function has all the required args

    The function can be a standalone function (class_method = False),
    or a class method (class_method = True).
    """
    func_module = load_attr(
        submission_module, func_name, attr_type="method" if class_method else "function"
    )
    if func_module is None:
        return 0

    score = 0
    try:
        sig = inspect.signature(func_module)
        actual_args = list(sig.parameters.keys())
    except Exception as e:
        logger.error(f"Error extracting parameters from function {func_name}: {e}")
        return 0

    required_args = list(func_config.get("required_args", {}).keys())
    if class_method:
        # Remove "self" from the list of actual / required args
        # as it's a default argument for class methods
        if "self" in actual_args:
            actual_args = [arg for arg in actual_args if arg != "self"]
        if "self" in required_args:
            required_args = [arg for arg in required_args if arg != "self"]

    if func_config.get("exact_match", False):
        score = int(actual_args == required_args)
    else:
        score = int(all(arg in actual_args for arg in required_args))
    return score


def get_type_str(type_class: type) -> str:
    """
    Get the string representation of a type class.
    """
    return f"{type_class.__module__}.{type_class.__name__}"


def grade_function_execution(
    submission_module: ModuleType,
    func_name: str,
    func_config: dict,
    class_method: bool = False,
) -> int:
    """
    A function is considered successfully executed if:
    - the function can be called with the provided arguments
    - if provided, the function output matches the expected type and shape

    The function can be a standalone function (class_method = False),
    or a class method (class_method = True).
    """
    func_module = load_attr(
        submission_module, func_name, attr_type="method" if class_method else "function"
    )
    if func_module is None:
        return 0

    func_args = func_config.get("required_args", {})
    try:
        outputs = func_module(**func_args)
    except Exception as e:
        logger.error(f"Error executing function {func_name}: {e}")
        outputs = None
        score = 0

    score = 0
    if outputs is not None:
        outputs_scores = {}
        if "expected_output_type" in func_config:
            # Check if the output type matches the expected output type
            expected_output_type = func_config["expected_output_type"]

            try:
                if type(outputs) == tuple:
                    actual_output_type = [get_type_str(type(output)) for output in outputs]
                else:
                    actual_output_type = get_type_str(type(outputs))
            except Exception as e:
                logger.error(f"Error extracting outputs type from function {func_name}: {e}")
                actual_output_type = None

            outputs_scores["type"] = expected_output_type == actual_output_type

        if "expected_output_shape" in func_config:
            # Check if the output shape matches the expected output shape
            expected_output_shape = func_config["expected_output_shape"]

            try:
                if type(outputs) == tuple:
                    actual_output_shape = [list(output.shape) for output in outputs]
                else:
                    actual_output_shape = list(outputs.shape)
            except Exception as e:
                logger.error(f"Error extracting outputs shape from function {func_name}: {e}")
                actual_output_shape = None

            outputs_scores["shape"] = expected_output_shape == actual_output_shape

        if len(outputs_scores) > 0:
            # Check if all the output types and shapes match the expected ones
            score = int(all(outputs_scores.values()))
        else:
            score = 1

    return score


def grade_class_method(submission_module: ModuleType, coding_config: dict) -> dict:
    """
    Each class method is graded based on:
    - whether the class method is defined correctly
    - whether the class method can be called with the required arguments
    (and its output shape matches the expected output shape if provided)

    A class method can only be valid if the class can be initialized correctly.
    """
    total_method_defined = 0
    total_method_executed = 0
    total_method_num = 0

    required_classes = coding_config.get("required_classes", {})
    for class_name, class_config in required_classes.items():
        class_obj = initialize_class(submission_module, class_name, class_config)

        required_methods = class_config.get("required_methods", {})
        for method_name, method_config in required_methods.items():
            if method_name == "__init__":
                # Skip the __init__ method as it is already checked in class initialization
                continue

            if class_obj is not None:
                score = grade_function_definition(
                    class_obj, method_name, method_config, class_method=True
                )
                total_method_defined += score

                score = grade_function_execution(
                    class_obj, method_name, method_config, class_method=True
                )
                total_method_executed += score

            total_method_num += 1

    scores = {
        "defined_class_methods": total_method_defined,
        "executed_class_methods": total_method_executed,
        "total_class_methods": total_method_num,
    }
    return scores


def grade_function(submission_module: ModuleType, coding_config: dict) -> dict:
    """
    Each standalone function is graded based on:
    - whether the function is defined correctly
    - whether the function can be called with the required arguments
    (and its output shape matches the expected output shape if provided)
    """
    total_function_defined = 0
    total_function_executed = 0
    total_function_num = 0

    required_functions = coding_config.get("required_functions", {})

    for func_name, func_config in required_functions.items():
        score = grade_function_definition(
            submission_module, func_name, func_config, class_method=False
        )
        total_function_defined += score

        score = grade_function_execution(
            submission_module, func_name, func_config, class_method=False
        )
        total_function_executed += score

        total_function_num += 1

    scores = {
        "defined_functions": total_function_defined,
        "executed_functions": total_function_executed,
        "total_functions": total_function_num,
    }
    return scores
