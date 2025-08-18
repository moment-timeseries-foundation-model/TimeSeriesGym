import ast
import os
import re
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
    Grade the feature enhancement challenge based on three criteria:
    1. Traceability and logging with TensorBoard
    2. Code quality and documentation
    3. Hydra configuration management

    Args:
        submission_folder_path: Path to the submission folder
        answers: DataFrame containing the true labels
        feature_enhancement_config: Configuration for feature enhancement
        solution_files_path: Path to solution files

    Returns:
        dict: Scores for each criterion and overall score
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

    # 1. Evaluate TensorBoard traceability (25%)
    logger.info("\n----- Evaluating TensorBoard Usage (25% of total score) -----")
    tensorboard_score = evaluate_tensorboard_usage(training_code_file, solution_files_path)

    # 2. Evaluate code quality and documentation (25%)
    logger.info("\n----- Evaluating Code Quality (25% of total score) -----")
    code_quality_score = evaluate_code_quality(model_code_file, training_code_file)

    # 3. Evaluate Hydra configuration usage (25%)
    logger.info("\n----- Evaluating Hydra Configuration Usage (25% of total score) -----")
    hydra_score = evaluate_hydra_usage(training_code_file, config_file)

    # 4. Calculate model accuracy (25%)
    accuracy = grade(submission, answers)
    logger.info("\n----- Evaluating Model Accuracy (25% of total score) -----")
    logger.info(f"[ACCURACY] Model prediction accuracy: {accuracy:.4f}/1.0")

    # Calculate overall score (weighted average)
    overall_score = (tensorboard_score + code_quality_score + hydra_score + accuracy) / 4

    scores = {
        "tensorboard_score": tensorboard_score,
        "code_quality_score": code_quality_score,
        "hydra_score": hydra_score,
        "accuracy": accuracy,
        "overall_score": overall_score,
    }

    logger.info("\n===== EVALUATION SUMMARY =====")
    logger.info(f"TensorBoard score: {tensorboard_score:.2f}/1.0 (25% weight)")
    logger.info(f"Code quality score: {code_quality_score:.2f}/1.0 (25% weight)")
    logger.info(f"Hydra config score: {hydra_score:.2f}/1.0 (25% weight)")
    logger.info(f"Model accuracy: {accuracy:.4f} (25% weight)")
    logger.info(f"Overall score: {overall_score:.2f}/1.0")

    # Print improvement suggestions
    logger.info("\n===== IMPROVEMENT SUGGESTIONS =====")

    if tensorboard_score < 1.0:
        logger.info("[SUGGESTION] To improve TensorBoard score:")
        if not re.search(
            r"from\s+torch\.utils\.tensorboard\s+import\s+SummaryWriter", training_code_file
        ):
            logger.info("  - Import SummaryWriter from torch.utils.tensorboard")
        if not re.search(r"SummaryWriter\s*\(", training_code_file):
            logger.info("  - Initialize SummaryWriter with an appropriate log directory")
        if not re.search(r"\.add_scalar\s*\(", training_code_file):
            logger.info("  - Log training and validation metrics using writer.add_scalar()")
        if not re.search(r"\.add_graph\s*\(", training_code_file):
            logger.info("  - Log model architecture using writer.add_graph()")

    if code_quality_score < 1.0:
        logger.info("[SUGGESTION] To improve code quality score:")
        logger.info(
            "  - Ensure all functions and classes have proper docstrings with Args/Returns sections"
        )
        logger.info("  - Add type hints to all function parameters and return values")
        logger.info("  - Follow PEP 8 guidelines for code formatting")
        logger.info("  - Run flake8 with '--max-line-length=100' to check for PEP 8 violations")

    if hydra_score < 1.0:
        logger.info("[SUGGESTION] To improve Hydra usage score:")
        if not re.search(r"import\s+hydra", training_code_file):
            logger.info("  - Import hydra and use the @hydra.main decorator")
        if not re.search(r"OmegaConf|DictConfig", training_code_file):
            logger.info("  - Use OmegaConf or DictConfig for configuration management")
        if not re.search(r"cfg\.model|cfg\['model'\]", training_code_file):
            logger.info("  - Access model parameters from the config object")
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

    # Check for TensorBoard import
    has_tb_import = re.search(
        r"from\s+torch\.utils\.tensorboard\s+import\s+SummaryWriter", training_file
    ) or re.search(r"import\s+torch\.utils\.tensorboard", training_file)

    max_score += 0.2
    if has_tb_import:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ TensorBoard SummaryWriter is properly imported: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ Missing TensorBoard import: 0.0/0.2")

    # Check for SummaryWriter initialization
    has_writer_init = re.search(r"SummaryWriter\s*\(", training_file)
    max_score += 0.2
    if has_writer_init:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ SummaryWriter is initialized: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ SummaryWriter is not initialized: 0.0/0.2")

    # Check for logging metrics with add_scalar
    has_add_scalar = re.search(r"\.add_scalar\s*\(", training_file)
    max_score += 0.2
    if has_add_scalar:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ Metrics are logged with add_scalar: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ Metrics are not logged with add_scalar: 0.0/0.2")

    # Check for model graph logging
    has_add_graph = re.search(r"\.add_graph\s*\(", training_file)
    max_score += 0.2
    if has_add_graph:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ Model graph is logged with add_graph: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ Model graph is not logged with add_graph: 0.0/0.2")

    # Check if there are actual TensorBoard logs in the submission
    tensorboard_log_dirs = ["runs", "logs", "tb_logs"]
    has_logs = False
    for log_dir in tensorboard_log_dirs:
        log_path = solution_path / log_dir
        if log_path.exists() and any(log_path.iterdir()):
            has_logs = True
            log_files = list(log_path.glob("**/*"))
            logger.info(f"[TENSORBOARD] Found {len(log_files)} log files in {log_dir}/")
            break

    max_score += 0.2
    if has_logs:
        score += 0.2
        logger.info("[TENSORBOARD] ✓ TensorBoard log files are present: 0.2/0.2")
    else:
        logger.info("[TENSORBOARD] ✗ No TensorBoard log files found: 0.0/0.2")
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
        model_docstring = False
        if len(model_ast.body) > 0 and isinstance(model_ast.body[0], ast.Expr):
            if hasattr(model_ast.body[0].value, "value"):  # Python 3.8+ (ast.Constant)
                if isinstance(model_ast.body[0].value, ast.Constant) and isinstance(
                    model_ast.body[0].value.value, str
                ):
                    model_docstring = True
                    score += 0.1
            elif isinstance(model_ast.body[0].value, ast.Str):  # Older Python versions
                model_docstring = True
                score += 0.1

        max_score += 0.1
        if model_docstring:
            logger.info("[CODE QUALITY] ✓ Model file has module docstring: 0.1/0.1")
        else:
            logger.info("[CODE QUALITY] ✗ Model file missing module docstring: 0.0/0.1")

        training_docstring = False
        if len(training_ast.body) > 0 and isinstance(training_ast.body[0], ast.Expr):
            if hasattr(training_ast.body[0].value, "value"):  # Python 3.8+ (ast.Constant)
                if isinstance(training_ast.body[0].value, ast.Constant) and isinstance(
                    training_ast.body[0].value.value, str
                ):
                    training_docstring = True
                    score += 0.1
            elif isinstance(training_ast.body[0].value, ast.Str):  # Older Python versions
                training_docstring = True
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
        has_args_returns = has_docstring_sections(
            model_file, ["Args", "Returns"]
        ) or has_docstring_sections(training_file, ["Args", "Returns"])
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
    """Check if AST has type hints in function definitions."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check return annotation
            if node.returns:
                return True

            # Check arguments
            for arg in node.args.args:
                if arg.annotation:
                    return True
    return False


def has_class_docstrings(tree):
    """Check if classes have docstrings."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if first statement in class body is a docstring
            if node.body and isinstance(node.body[0], ast.Expr):
                if hasattr(node.body[0].value, "value"):  # Python 3.8+ (ast.Constant)
                    if isinstance(node.body[0].value, ast.Constant) and isinstance(
                        node.body[0].value.value, str
                    ):
                        return True
                elif isinstance(node.body[0].value, ast.Str):  # Older Python versions
                    return True
    return False


def has_function_docstrings(tree):
    """Check if functions have docstrings."""
    has_docs = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip very small helper functions (likely don't need docs)
            if len(node.body) <= 2:
                continue

            # Check if first statement in function body is a docstring
            if node.body and isinstance(node.body[0], ast.Expr):
                if hasattr(node.body[0].value, "value"):  # Python 3.8+ (ast.Constant)
                    if isinstance(node.body[0].value, ast.Constant) and isinstance(
                        node.body[0].value.value, str
                    ):
                        has_docs = True
                    elif len(node.body) > 3:  # More than a tiny function without docs
                        return False
                elif isinstance(node.body[0].value, ast.Str):  # Older Python versions
                    has_docs = True
                else:
                    # If we find a substantial function without docs, return False
                    if len(node.body) > 3:  # More than a tiny function
                        return False
    return has_docs


def has_docstring_sections(code, sections):
    """Check if docstrings contain specific sections."""
    for section in sections:
        if f"{section}:" in code:
            return True
    return False


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
            ["flake8", "--max-line-length=100", model_file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        flake8_training = subprocess.run(
            ["flake8", "--max-line-length=100", training_file_path],
            capture_output=True,
            text=True,
            check=False,
        )

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
        linting_details.append(f"✗ Failed to run linting tools: {e}")
        # If linting tools fail, give partial credit
        linting_score = 0.5

    return linting_score, linting_details


def evaluate_hydra_usage(training_file: str, config_file: str) -> float:
    """
    Evaluate proper usage of Hydra for configuration management.

    Args:
        training_file: The training script content
        config_file: The Hydra config file content

    Returns:
        float: Score between 0 and 1
    """
    score = 0.0
    max_score = 0.0

    logger.info("[HYDRA] Starting Hydra usage evaluation")

    # Check for Hydra import
    has_hydra_import = re.search(r"import\s+hydra", training_file) or re.search(
        r"from\s+hydra", training_file
    )
    max_score += 0.2
    if has_hydra_import:
        score += 0.2
        logger.info("[HYDRA] ✓ Hydra is properly imported: 0.2/0.2")
    else:
        logger.info("[HYDRA] ✗ Missing Hydra import: 0.0/0.2")

    # Check for @hydra.main decorator
    has_hydra_decorator = re.search(r"@hydra\.main", training_file)
    max_score += 0.2
    if has_hydra_decorator:
        score += 0.2
        logger.info("[HYDRA] ✓ @hydra.main decorator is used: 0.2/0.2")
    else:
        logger.info("[HYDRA] ✗ Missing @hydra.main decorator: 0.0/0.2")

    # Check for OmegaConf or DictConfig usage
    has_config_usage = re.search(r"OmegaConf|DictConfig", training_file)
    max_score += 0.2
    if has_config_usage:
        score += 0.2
        logger.info("[HYDRA] ✓ OmegaConf/DictConfig is used: 0.2/0.2")
    else:
        logger.info("[HYDRA] ✗ Missing OmegaConf or DictConfig usage: 0.0/0.2")

    # Check if config is used for model parameters
    has_model_config = re.search(r"cfg\.model|cfg\['model'\]", training_file)
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
    print(f"Overall Score: {scores['overall_score']:.2f}/1.0")
