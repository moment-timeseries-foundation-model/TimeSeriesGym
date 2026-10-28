import ast
import os
import re
import struct
import subprocess
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score

from timeseriesgym.utils import get_logger, load_scripts, load_submissions

logger = get_logger(__name__)


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Grade the model predictions using accuracy score."""
    return accuracy_score(y_true=submission["label"], y_pred=answers["label"])


def grade_feature_enhancement(
    submission_folder_path: Path,
    answers: DataFrame,
    feature_enhancement_config: dict,
    solution_files_path: Path | None = None,
) -> dict:
    """
    Grade the feature enhancement challenge: the score is the model's test accuracy, gated by a
    rubric of three engineering criteria:
    1. Traceability and logging with TensorBoard
    2. Code quality and documentation
    3. Hydra configuration management

    Every check of the three criteria must pass (each criterion then scores 1.0), or the score
    is 0. The criteria's sub-scores are reported but do not move the score.

    Args:
        submission_folder_path: Path to the submission folder
        answers: DataFrame containing the true labels
        feature_enhancement_config: Configuration for feature enhancement
        solution_files_path: Path to solution files

    Returns:
        dict: Scores for each criterion, the accuracy, whether the rubric passed, and the
            overall score (the accuracy if the rubric passed, else 0)
    """
    submission = load_submissions(submission_folder_path / "submission.csv")

    logger.info("===== STARTING FEATURE ENHANCEMENT EVALUATION =====")

    # Get model files for inspection
    model_code_file = load_scripts(
        solution_files_path / feature_enhancement_config["model_code_file"]
    )
    training_code_file = load_scripts(
        solution_files_path / feature_enhancement_config["training_file"]
    )
    config_file = None
    try:
        config_path = solution_files_path / "conf" / feature_enhancement_config["config_file"]
        if config_path.exists():
            with open(config_path) as f:
                config_file = f.read()
    except Exception as e:
        logger.warning(f"Failed to read config file: {e}")

    # 1. Evaluate TensorBoard traceability
    logger.info("\n----- Evaluating TensorBoard Usage -----")
    tensorboard_score = evaluate_tensorboard_usage(training_code_file, solution_files_path)

    # 2. Evaluate code quality and documentation
    logger.info("\n----- Evaluating Code Quality -----")
    code_quality_score = evaluate_code_quality(model_code_file, training_code_file)

    # 3. Evaluate Hydra configuration usage
    logger.info("\n----- Evaluating Hydra Configuration Usage -----")
    hydra_score = evaluate_hydra_usage(training_code_file, config_file, model_code_file)

    # 4. Calculate model accuracy
    accuracy = grade(submission, answers)
    logger.info("\n----- Evaluating Model Accuracy -----")
    logger.info(f"[ACCURACY] Model prediction accuracy: {accuracy:.4f}/1.0")

    # The score is the accuracy if every check passes, else 0. It was the mean of the three
    # criteria and the accuracy, so 75% of it came from keyword and AST checks: with the rubric
    # passed, predicting the most frequent class for every ECG scored 0.888. A criterion's score
    # is the summed weight of its passed checks over the summed weight of all of them, added in
    # the same order, so it is exactly 1.0 if and only if all of them pass (lint: no problems).
    rubric_passed = tensorboard_score == code_quality_score == hydra_score == 1.0
    overall_score = accuracy if rubric_passed else 0.0

    scores = {
        "tensorboard_score": tensorboard_score,
        "code_quality_score": code_quality_score,
        "hydra_score": hydra_score,
        "accuracy": accuracy,
        "rubric_passed": rubric_passed,
        "overall_score": overall_score,
    }

    logger.info("\n===== EVALUATION SUMMARY =====")
    logger.info(f"TensorBoard score: {tensorboard_score:.2f}/1.0")
    logger.info(f"Code quality score: {code_quality_score:.2f}/1.0")
    logger.info(f"Hydra config score: {hydra_score:.2f}/1.0")
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info(f"Every rubric check passed: {rubric_passed}")
    logger.info(f"Overall score (the accuracy if every check passed, else 0): {overall_score:.4f}")

    # Print improvement suggestions
    logger.info("\n===== IMPROVEMENT SUGGESTIONS =====")

    if not rubric_passed:
        logger.info("[SUGGESTION] The score is 0 until every check marked ✗ above passes")

    training_tree = _parse(training_code_file)
    if tensorboard_score < 1.0:
        logger.info("[SUGGESTION] To improve TensorBoard score:")
        if not imports_summary_writer(training_tree):
            logger.info("  - Import SummaryWriter from torch.utils.tensorboard")
        if not calls_function(training_tree, "SummaryWriter"):
            logger.info("  - Initialize SummaryWriter with an appropriate log directory")
        if not calls_method(training_tree, "add_scalar"):
            logger.info("  - Log training and validation metrics using writer.add_scalar()")
        if not calls_method(training_tree, "add_graph"):
            logger.info("  - Log model architecture using writer.add_graph()")

    if code_quality_score < 1.0:
        logger.info("[SUGGESTION] To improve code quality score:")
        logger.info(
            "  - Ensure all functions and classes have proper docstrings with Args/Returns sections"
        )
        logger.info("  - Add type hints to all function parameters and return values")
        logger.info("  - Follow PEP 8 guidelines for code formatting")
        logger.info(f"  - Run `flake8 {' '.join(FLAKE8_OPTIONS)}` to check for PEP 8 violations")

    if hydra_score < 1.0:
        logger.info("[SUGGESTION] To improve Hydra usage score:")
        if not (imports_hydra(training_tree) and decorated_with_hydra_main(training_tree)):
            logger.info("  - Import hydra and use the @hydra.main decorator")
        if not uses_omegaconf(training_tree):
            logger.info("  - Use OmegaConf or DictConfig for configuration management")
        if not builds_model_from_config(training_code_file, model_code_file):
            logger.info("  - Build the model from the config's model section (e.g. cfg.model)")
        if not config_file or "model:" not in config_file:
            logger.info("  - Create a proper YAML config file with model parameters")

    if accuracy < 0.95:
        logger.info("[SUGGESTION] To improve model accuracy:")
        logger.info("  - Try different model architectures or hyperparameters")
        logger.info("  - Implement data augmentation or feature engineering")
        logger.info("  - Use learning rate scheduling")
        logger.info("  - Add regularization to prevent overfitting")

    return scores


def evaluate_tensorboard_usage(training_file: str, solution_path: Path) -> float:
    """
    Evaluate the proper usage of TensorBoard for experiment tracking.

    Args:
        training_file: The training script content
        solution_path: Path to the solution folder

    Returns:
        float: Score between 0 and 1
    """
    score = 0.0
    max_score = 0.0

    logger.info("[TENSORBOARD] Starting TensorBoard usage evaluation")
    # The code is parsed, so comments and string literals earn nothing.
    training_tree = _parse(training_file)

    # Check for TensorBoard import
    has_tb_import = imports_summary_writer(training_tree)

    max_score += 0.2
    if has_tb_import:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ TensorBoard SummaryWriter is properly imported: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ Missing TensorBoard import: 0.0/0.2")

    # Check for SummaryWriter initialization
    has_writer_init = calls_function(training_tree, "SummaryWriter")
    max_score += 0.2
    if has_writer_init:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ SummaryWriter is initialized: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ SummaryWriter is not initialized: 0.0/0.2")

    # Check for logging metrics with add_scalar
    has_add_scalar = calls_method(training_tree, "add_scalar")
    max_score += 0.2
    if has_add_scalar:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ Metrics are logged with add_scalar: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ Metrics are not logged with add_scalar: 0.0/0.2")

    # Check for model graph logging
    has_add_graph = calls_method(training_tree, "add_graph")
    max_score += 0.2
    if has_add_graph:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ Model graph is logged with add_graph: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ Model graph is not logged with add_graph: 0.0/0.2")

    # Check if there are actual TensorBoard logs in the submission: an event file that holds
    # logged scalars, not merely a file with an event file's name
    tensorboard_log_dirs = ["runs", "logs", "tb_logs"]
    event_file = None
    for log_dir in tensorboard_log_dirs:
        log_path = solution_path / log_dir
        if log_path.is_dir() and not log_path.is_symlink():
            event_file = find_event_file(log_path)
            if event_file is not None:
                logger.info(f"[TENSORBOARD] Found an event file with scalars: {event_file}")
                break

    max_score += 0.2
    if event_file is not None:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ TensorBoard log files are present: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ No TensorBoard event file with scalars found: 0.0/0.2")
        logger.info("[TENSORBOARD] Searched directories: " + ", ".join(tensorboard_log_dirs))

    logger.info(f"[TENSORBOARD] Final score: {score:.2f}/{max_score:.2f}")

    # Normalize the score to ensure it can reach 1.0
    if max_score > 0:
        score = score / max_score
    else:
        score = 0.0

    return score


def evaluate_code_quality(model_file: str, training_file: str) -> float:
    """
    Evaluate the code quality and documentation using proper code analysis.

    This function analyzes:
    1. Proper docstrings and documentation
    2. Type hints usage
    3. Code structure and organization
    4. Code style and linting compliance (PEP 8)
    5. Naming conventions

    Args:
        model_file: The model code content
        training_file: The training script content

    Returns:
        float: Score between 0 and 1
    """
    import os
    import tempfile

    score = 0.0
    max_score = 0.0

    logger.info("[CODE QUALITY] Starting code quality evaluation")

    # Create temporary files for the code
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as model_temp:
        model_temp.write(model_file)
        model_temp_name = model_temp.name

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as training_temp:
        training_temp.write(training_file)
        training_temp_name = training_temp.name

    try:
        # Parse the code with ast
        try:
            model_ast = ast.parse(model_file)
            training_ast = ast.parse(training_file)
            # Successfully parsed - basic syntax check passes
            score += 0.1
            max_score += 0.1
            logger.info("[CODE QUALITY] ✓ Syntax check passed: 0.1/0.1")
        except SyntaxError:
            # Failed basic syntax check
            max_score += 0.1
            logger.warning("[CODE QUALITY] ✗ Code contains syntax errors: 0.0/0.1")
            return score

        # Check for module-level docstrings
        model_docstring = ast.get_docstring(model_ast) is not None
        if model_docstring:
            score += 0.1

        max_score += 0.1
        if model_docstring:
            logger.info("[CODE QUALITY] ✓ Model file has module docstring: 0.1/0.1")
        else:
            logger.info("[CODE QUALITY] ✗ Model file missing module docstring: 0.0/0.1")

        training_docstring = ast.get_docstring(training_ast) is not None
        if training_docstring:
            score += 0.1

        max_score += 0.1
        if training_docstring:
            logger.info("[CODE QUALITY] ✓ Training file has module docstring: 0.1/0.1")
        else:
            logger.info("[CODE QUALITY] ✗ Training file missing module docstring: 0.0/0.1")

        # Check for type annotations
        type_hints_model = has_type_hints(model_ast)
        type_hints_training = has_type_hints(training_ast)

        max_score += 0.1
        if type_hints_model and type_hints_training:
            score += 0.1
            logger.info("[CODE QUALITY] ✓ Both files have type annotations: 0.1/0.1")
        else:
            hint_issues = []
            if not type_hints_model:
                hint_issues.append("model")
            if not type_hints_training:
                hint_issues.append("training")
            logger.info(
                f"[CODE QUALITY] ✗ Missing type hints in {', '.join(hint_issues)} files: 0.0/0.1"
            )

        # Check for function/class docstrings
        class_docs = has_class_docstrings(model_ast)
        func_docs_model = has_function_docstrings(model_ast)
        func_docs_training = has_function_docstrings(training_ast)

        max_score += 0.05
        if class_docs:
            score += 0.05
            logger.info("[CODE QUALITY] ✓ Model classes have docstrings: 0.05/0.05")
        else:
            logger.info("[CODE QUALITY] ✗ Model classes missing docstrings: 0.0/0.05")

        max_score += 0.1
        if func_docs_model and func_docs_training:
            score += 0.1
            logger.info("[CODE QUALITY] ✓ Functions have docstrings: 0.1/0.1")
        else:
            func_issues = []
            if not func_docs_model:
                func_issues.append("model")
            if not func_docs_training:
                func_issues.append("training")
            logger.info(
                f"[CODE QUALITY] ✗ Missing func docstring in {', '.join(func_issues)} file: 0.0/0.1"
            )

        # Check for docstring format (Args/Returns sections)
        has_args_returns = has_docstring_sections([model_ast, training_ast], ["Args", "Returns"])
        max_score += 0.1
        if has_args_returns:
            score += 0.1
            logger.info("[CODE QUALITY] ✓ Docstrings have Args/Returns sections: 0.1/0.1")
        else:
            logger.info("[CODE QUALITY] ✗ Docstrings missing Args/Returns sections: 0.0/0.1")

        # Run flake8 and pycodestyle to check for PEP 8 compliance
        max_score += 0.25
        linting_score, linting_details = check_linting(model_temp_name, training_temp_name)
        linting_weighted = linting_score * 0.25
        score += linting_weighted
        logger.info(f"[CODE QUALITY] Linting score: {linting_weighted:.2f}/0.25")

        for detail in linting_details:
            logger.info(f"[CODE QUALITY] {detail}")

    finally:
        # Clean up temporary files
        try:
            os.unlink(model_temp_name)
            os.unlink(training_temp_name)
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")

    logger.info(f"[CODE QUALITY] Final score: {score:.2f}/{max_score:.2f}")

    # Normalize the score to ensure it can reach 1.0
    if max_score > 0:
        score = score / max_score
    else:
        score = 0.0

    return score


def has_type_hints(tree):
    """Check that every function annotates its parameters (except self/cls) and its return; an
    `__init__` may leave out `-> None`, as PEP 484 allows."""
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    for node in functions:
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        params = [arg for arg in params if arg is not None and arg.arg not in ("self", "cls")]
        no_return = node.returns is None and node.name != "__init__"
        if no_return or any(arg.annotation is None for arg in params):
            return False
    return bool(functions)


def has_class_docstrings(tree):
    """Check that every class has a docstring."""
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return bool(classes) and all(ast.get_docstring(node) is not None for node in classes)


def has_function_docstrings(tree):
    """Check if functions have docstrings."""
    has_docs = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Skip very small helper functions (likely don't need docs)
            if len(node.body) <= 2:
                continue

            if ast.get_docstring(node) is not None:
                has_docs = True
            elif len(node.body) > 3:  # More than a tiny function without docs
                return False
    return has_docs


def has_docstring_sections(trees, sections):
    """Check that each section appears in a docstring (not in comments or code)."""
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings = [
        ast.get_docstring(node) or ""
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, documented)
    ]
    return all(any(f"{section}:" in doc for doc in docstrings) for section in sections)


# A YAML comment: `#` at the start of a line or after whitespace, outside quoted strings (group 1,
# which is kept: `name: "run #1"` is not a comment).
YAML_COMMENT = re.compile(r"""("(?:[^"\\\n]|\\.)*"|'(?:[^'\n]|'')*')|(?:^|(?<=\s))#.*""", re.M)


# The checks of train.py's code use its syntax tree, so that comments and string literals do not
# count (as in the rl_envs rubric, rl_envs/tsgym_rl/grader/fe_rubric.py). A tree that is None (the
# file does not parse) passes none of them.


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _walk(tree: ast.AST | None):
    return ast.walk(tree) if tree is not None else ()


def calls_method(tree: ast.AST | None, name: str) -> bool:
    """A call `<expression>.<name>(...)`, e.g. `writer.add_scalar(...)`."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        for node in _walk(tree)
    )


def calls_function(tree: ast.AST | None, name: str) -> bool:
    """A call `<name>(...)` or `<expression>.<name>(...)`."""
    return any(isinstance(node, ast.Call) and _name_of(node.func) == name for node in _walk(tree))


def imports_summary_writer(tree: ast.AST | None) -> bool:
    """`from torch.utils.tensorboard import SummaryWriter` or `import torch.utils.tensorboard`."""
    for node in _walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "torch.utils.tensorboard":
            if node.level == 0 and any(alias.name == "SummaryWriter" for alias in node.names):
                return True
        if isinstance(node, ast.Import) and any(
            alias.name == "torch.utils.tensorboard"
            or alias.name.startswith("torch.utils.tensorboard.")
            for alias in node.names
        ):
            return True
    return False


def imports_hydra(tree: ast.AST | None) -> bool:
    """`import hydra` (or a submodule), or `from hydra... import ...`."""
    for node in _walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "hydra" or alias.name.startswith("hydra.") for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "hydra" or node.module.startswith("hydra."):
                return True
    return False


def decorated_with_hydra_main(tree: ast.AST | None) -> bool:
    """A function decorated with `@hydra.main` or `@hydra.main(...)`."""
    for node in _walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "main"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "hydra"
                ):
                    return True
    return False


def uses_omegaconf(tree: ast.AST | None) -> bool:
    """The name `OmegaConf` or `DictConfig`, imported or used."""
    names = {"OmegaConf", "DictConfig"}
    return any(
        _name_of(node) in names
        or (isinstance(node, ast.alias) and node.name.rsplit(".", 1)[-1] in names)
        for node in _walk(tree)
    )


# A TensorBoard event file is a TFRecord file: records of a uint64 length, the masked CRC32C of
# the length, the data and the masked CRC32C of the data, each an Event protobuf (event.proto).
# The reader below is the rl_envs rubric's.
EVENT_FILE_PREFIX = "events.out.tfevents."
MAX_EVENT_FILE_BYTES = 16 * 1024**2  # read per event file
MAX_EVENT_FILES = 100  # checked per log directory


def _crc32c_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC32C_TABLE = _crc32c_table()


def masked_crc32c(data: bytes) -> int:
    """TFRecord's masked CRC32C (Castagnoli) checksum."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    crc ^= 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if i >= len(buf) or shift > 63:
            raise ValueError("bad varint")
        value |= (buf[i] & 0x7F) << shift
        shift += 7
        i += 1
        if not buf[i - 1] & 0x80:
            return value, i


def _protobuf_fields(buf: bytes) -> list[tuple[int, int, int | bytes]]:
    """(field number, wire type, value) of a protobuf message: an int for varint and fixed-size
    fields, bytes for length-delimited ones. Raises ValueError if it is not a message."""
    fields, i = [], 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            if i + size > len(buf):
                raise ValueError("truncated field")
            value, i = int.from_bytes(buf[i : i + size], "little"), i + size
        elif wire == 2:
            size, i = _varint(buf, i)
            if i + size > len(buf):
                raise ValueError("truncated field")
            value, i = buf[i : i + size], i + size
        else:
            raise ValueError(f"wire type {wire}")
        fields.append((number, wire, value))
    return fields


def _messages(fields: list, number: int) -> list[bytes]:
    return [value for n, wire, value in fields if n == number and wire == 2]


def _has_scalar(summary: bytes) -> bool:
    """Whether a Summary holds a scalar: a value with a `simple_value` (as `add_scalar` writes
    it), or one whose metadata names the `scalars` plugin."""
    for value in _messages(_protobuf_fields(summary), 1):  # repeated Value value = 1
        fields = _protobuf_fields(value)
        if any(n == 2 and wire == 5 for n, wire, _ in fields):  # float simple_value = 2
            return True
        for metadata in _messages(fields, 9):  # SummaryMetadata metadata = 9
            for plugin in _messages(_protobuf_fields(metadata), 1):  # PluginData plugin_data = 1
                if b"scalars" in _messages(_protobuf_fields(plugin), 1):  # plugin_name = 1
                    return True
    return False


def is_event_file_with_scalars(data: bytes) -> bool:
    """Whether `data` is a TensorBoard event file that holds logged scalars: valid TFRecord
    records, the first an Event whose file_version starts with `brain.Event:`, a later one an
    Event whose summary holds a scalar."""
    i, first = 0, True
    try:
        while i + 12 <= len(data):
            (length,) = struct.unpack_from("<Q", data, i)
            if masked_crc32c(data[i : i + 8]) != struct.unpack_from("<I", data, i + 8)[0]:
                return False
            start, end = i + 12, i + 12 + length
            if end + 4 > len(data):
                return False
            record = data[start:end]
            if masked_crc32c(record) != struct.unpack_from("<I", data, end)[0]:
                return False
            fields = _protobuf_fields(record)
            if first:
                versions = _messages(fields, 3)  # string file_version = 3
                if not versions or not versions[0].startswith(b"brain.Event:"):
                    return False
                first = False
            elif any(_has_scalar(summary) for summary in _messages(fields, 5)):  # summary = 5
                return True
            i = end + 4
    except ValueError:
        return False
    return False


def find_event_file(log_dir: Path) -> Path | None:
    """The first file named `events.out.tfevents.*` under `log_dir` (symbolic links are not
    followed) that is an event file with logged scalars, or None."""
    candidates = sorted(log_dir.rglob(f"{EVENT_FILE_PREFIX}*"))
    candidates = [path for path in candidates if path.is_file() and not path.is_symlink()]
    for path in candidates[:MAX_EVENT_FILES]:
        try:
            with open(path, "rb") as f:
                data = f.read(MAX_EVENT_FILE_BYTES)
        except OSError:
            continue
        if is_event_file_with_scalars(data):
            return path
    return None


def _parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None


def _model_section_owner(node: ast.AST) -> ast.AST | None:
    """The object whose `model` section `node` reads (`x.model`, `x["model"]` or
    `getattr(x, "model")`), else None. `self.model` is an attribute, not a config section."""
    if isinstance(node, ast.Attribute) and node.attr == "model":
        owner = node.value
    elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        owner = node.value if node.slice.value == "model" else None
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "model"
    ):
        owner = node.args[0]
    else:
        return None
    return None if isinstance(owner, ast.Name) and owner.id in ("self", "cls") else owner


def _reads_a_parameters_model_section(definition: ast.AST) -> bool:
    """Whether a function, or a method of a class, reads the model section of a parameter."""
    for function in ast.walk(definition):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = function.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        names = {arg.arg for arg in params if arg is not None} - {"self", "cls"}
        for node in ast.walk(function):
            owner = _model_section_owner(node)
            if isinstance(owner, ast.Name) and owner.id in names:
                return True
    return False


def builds_model_from_config(training_file: str, model_file: str = "") -> bool:
    """
    Check that train.py builds the model from the config's `model` section: it passes the
    section, or a value read from it, to a class or function of model.py or to Hydra's
    `instantiate`, or it passes an argument to a class or function of model.py that reads the
    model section of its parameter (as `ECGClassifier(cfg)`, whose `__init__` reads
    `config.model`). Printing or only mentioning `cfg.model` does not count.

    Args:
        training_file: The training script content
        model_file: The model code content

    Returns:
        bool: Whether the model is built from the config's model section
    """
    training_tree, model_tree = _parse(training_file), _parse(model_file)
    if training_tree is None:
        return False
    definitions = {
        node.name: node
        for node in (model_tree.body if model_tree is not None else [])
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    aliases = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(training_tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    # names bound to the model section or to values read from it (`model_cfg = cfg.model`)
    section_names: set[str] = set()

    def reads_section(expression: ast.AST) -> bool:
        return any(
            _model_section_owner(node) is not None
            or (isinstance(node, ast.Name) and node.id in section_names)
            for node in ast.walk(expression)
        )

    assignments = [n for n in ast.walk(training_tree) if isinstance(n, ast.Assign | ast.AnnAssign)]
    for _ in range(3):
        for node in assignments:
            if node.value is not None and reads_section(node.value):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                section_names.update(
                    name.id for t in targets for name in ast.walk(t) if isinstance(name, ast.Name)
                )

    for call in ast.walk(training_tree):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Name):
            name = aliases.get(func.id, func.id)
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
        if (name in definitions or name == "instantiate") and any(map(reads_section, arguments)):
            return True
        if name in definitions and arguments:
            if _reads_a_parameters_model_section(definitions[name]):
                return True
    return False


# The lint check's flake8 options. No configuration file (flake8 reads one from its working
# directory) and no `# noqa` comment can turn a check off. E203 (whitespace before ':') is
# ignored, as black and ruff recommend: their slices, such as `x[i : i + 1]`, follow PEP 8.
FLAKE8_OPTIONS = ("--isolated", "--disable-noqa", "--extend-ignore=E203", "--max-line-length=100")


def check_linting(model_file_path: str, training_file_path: str) -> tuple[float, list[str]]:
    """
    Check code linting and style compliance with PEP 8.

    This function runs:
    - flake8: to check for PEP 8 compliance

    It evaluates:
    - Line length
    - Indentation
    - Whitespace
    - Import style
    - Naming conventions
    - Code formatting consistency
    - Other PEP 8 guidelines

    Args:
        model_file_path: Path to the model code file
        training_file_path: Path to the training code file

    Returns:
        tuple: (score, details) where:
            - score is a float between 0 and 1 based on linting compliance
            - details is a list of strings containing linting issues
    """
    linting_score = 1.0  # Start with perfect score
    max_errors = 20  # Maximum number of errors beyond which score is 0
    linting_details = []

    try:
        # Run flake8 to check for PEP 8 compliance
        flake8_model = subprocess.run(
            ["flake8", *FLAKE8_OPTIONS, model_file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        flake8_training = subprocess.run(
            ["flake8", *FLAKE8_OPTIONS, training_file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        for run in (flake8_model, flake8_training):
            if run.returncode != 0 and not run.stdout.strip():  # it failed, without a report
                raise subprocess.SubprocessError(run.stderr.strip()[-500:])

        # Count number of linting errors
        model_flake8_errors = flake8_model.stdout.strip().split("\n")
        if len(model_flake8_errors) == 1 and model_flake8_errors[0] == "":
            model_errors = 0
            linting_details.append("✓ Model file has no PEP 8 violations")
        else:
            model_errors = len(model_flake8_errors)
            # Add at most 5 errors to details
            num_errors_to_show = min(5, model_errors)
            linting_details.append(f"✗ Model file has {model_errors} PEP 8 violations")
            for i in range(num_errors_to_show):
                linting_details.append(f"  - {model_flake8_errors[i]}")
            if model_errors > num_errors_to_show:
                linting_details.append(f"  - ... and {model_errors - num_errors_to_show} more")

        training_flake8_errors = flake8_training.stdout.strip().split("\n")
        if len(training_flake8_errors) == 1 and training_flake8_errors[0] == "":
            training_errors = 0
            linting_details.append("✓ Training file has no PEP 8 violations")
        else:
            training_errors = len(training_flake8_errors)
            # Add at most 5 errors to details
            num_errors_to_show = min(5, training_errors)
            linting_details.append(f"✗ Training file has {training_errors} PEP 8 violations")
            for i in range(num_errors_to_show):
                linting_details.append(f"  - {training_flake8_errors[i]}")
            if training_errors > num_errors_to_show:
                linting_details.append(f"  - ... and {training_errors - num_errors_to_show} more")

        total_errors = model_errors + training_errors

        # If files have formatting issues, mention them
        if model_errors == 0:
            linting_details.append("✓ Model file follows style guidelines")
        else:
            linting_details.append("✗ Model file has formatting issues")

        if training_errors == 0:
            linting_details.append("✓ Training file follows style guidelines")
        else:
            linting_details.append("✗ Training file has formatting issues")

        # Reduce score based on number of errors
        if total_errors > 0:
            # Penalize score proportionally to number of errors
            linting_score = max(0, 1.0 - (total_errors / max_errors))
            linting_details.append(
                f"Linting issues cost {(1.0 - linting_score) * 100:.1f}% of available points"
            )

    except (subprocess.SubprocessError, FileNotFoundError) as e:
        # The score is 0 unless the lint check passes, so a missing tool must not look like a
        # failed check (it used to give partial credit, 0.5).
        raise RuntimeError(
            f"flake8 is needed to grade this challenge and failed to run: {e}"
        ) from e

    return linting_score, linting_details


def evaluate_hydra_usage(training_file: str, config_file: str, model_file: str = "") -> float:
    """
    Evaluate proper usage of Hydra for configuration management.

    Args:
        training_file: The training script content
        config_file: The Hydra config file content
        model_file: The model code content (its classes and functions build the model)

    Returns:
        float: Score between 0 and 1
    """
    score = 0.0
    max_score = 0.0

    logger.info("[HYDRA] Starting Hydra usage evaluation")
    # The code is parsed, so comments and string literals earn nothing.
    training_tree = _parse(training_file)
    if config_file:
        config_file = YAML_COMMENT.sub(lambda m: m.group(1) or "", config_file)

    # Check for Hydra import
    has_hydra_import = imports_hydra(training_tree)
    max_score += 0.2
    if has_hydra_import:
        score += 0.2
        logger.info("[HYDRA] ✓ Hydra is properly imported: 0.2/0.2")
    else:
        logger.info("[HYDRA] ✗ Missing Hydra import: 0.0/0.2")

    # Check for @hydra.main decorator
    has_hydra_decorator = decorated_with_hydra_main(training_tree)
    max_score += 0.2
    if has_hydra_decorator:
        score += 0.2
        logger.info("[HYDRA] ✓ @hydra.main decorator is used: 0.2/0.2")
    else:
        logger.info("[HYDRA] ✗ Missing @hydra.main decorator: 0.0/0.2")

    # Check for OmegaConf or DictConfig usage
    has_config_usage = uses_omegaconf(training_tree)
    max_score += 0.2
    if has_config_usage:
        score += 0.2
        logger.info("[HYDRA] ✓ OmegaConf/DictConfig is used: 0.2/0.2")
    else:
        logger.info("[HYDRA] ✗ Missing OmegaConf or DictConfig usage: 0.0/0.2")

    # Check if config is used for model parameters
    has_model_config = builds_model_from_config(training_file, model_file)
    max_score += 0.2
    if has_model_config:
        score += 0.2
        logger.info("[HYDRA] ✓ Config is used for model parameters: 0.2/0.2")
    else:
        logger.info("[HYDRA] ✗ Config is not used for model parameters: 0.0/0.2")

    # Check if a proper config file exists with model parameters
    has_config_file = False
    if config_file and "model:" in config_file:
        has_config_file = True
        score += 0.2

    max_score += 0.2
    if has_config_file:
        logger.info("[HYDRA] ✓ Config file exists with model parameters: 0.2/0.2")
    elif config_file:
        logger.info("[HYDRA] ✗ Config file exists but missing model section: 0.0/0.2")
    else:
        logger.info("[HYDRA] ✗ Missing config file: 0.0/0.2")

    logger.info(f"[HYDRA] Final score: {score:.2f}/{max_score:.2f}")

    # Normalize the score to ensure it can reach 1.0
    if max_score > 0:
        score = score / max_score
    else:
        score = 0.0

    return score


# For direct testing #
if __name__ == "__main__":
    import os
    from pathlib import Path

    import pandas as pd

    # Create test paths - use absolute path
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    example_dir = current_dir / "example"

    # Ensure example directory exists
    os.makedirs(example_dir, exist_ok=True)
    os.makedirs(example_dir / "conf", exist_ok=True)

    # Create a mock submission and answers for testing
    submission_path = example_dir / "submission.csv"
    if not submission_path.exists():
        mock_submission = pd.DataFrame({"label": [0, 1, 2, 3, 4] * 20})
        mock_submission.to_csv(submission_path, index=False)

    mock_answers = pd.DataFrame({"label": [0, 1, 2, 3, 4] * 20})

    # Config for testing
    test_config = {
        "model_code_file": "model.py",
        "training_file": "train.py",
        "config_file": "config.yaml",
    }

    print("Testing grade_feature_enhancement on example files...")
    scores = grade_feature_enhancement(
        submission_folder_path=example_dir,
        answers=mock_answers,
        feature_enhancement_config=test_config,
        solution_files_path=example_dir,
    )

    print("\n=== Grading Results ===")
    print(f"TensorBoard Score: {scores['tensorboard_score']:.2f}/1.0")
    print(f"Code Quality Score: {scores['code_quality_score']:.2f}/1.0")
    print(f"Hydra Config Score: {scores['hydra_score']:.2f}/1.0")
    print(f"Accuracy Score: {scores['accuracy']:.2f}/1.0")
    print(f"Rubric Passed: {scores['rubric_passed']}")
    print(f"Overall Score: {scores['overall_score']:.2f}/1.0")
