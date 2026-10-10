"""
LLM-as-a-judge evaluation for code reasonableness using DeepEval's GEval.

This module evaluates whether a submission made a genuine modeling attempt
rather than hallucinating an output that matches the submission format.
"""

import os
import re
from dataclasses import dataclass

# deepeval starts telemetry when it is imported; opt out before the import (as in judge.py)
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import GEval  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from timeseriesgym.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class ReasonablenessResult:
    """Result of a reasonableness evaluation."""

    is_reasonable: bool
    score: float
    threshold: float
    has_suspicious_patterns: bool
    has_modeling_patterns: bool
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "is_reasonable": self.is_reasonable,
            "score": self.score,
            "threshold": self.threshold,
            "has_suspicious_patterns": self.has_suspicious_patterns,
            "has_modeling_patterns": self.has_modeling_patterns,
            "reason": self.reason,
        }


def has_suspicious_patterns(code: str) -> bool:
    """Quick regex check for obvious non-genuine patterns."""
    patterns = [
        # Directly reading sample submission
        r'pd\.read_csv\([\'"].*sample_submission',
        r'sample_submission.*\.read|read.*sample_submission',
        # Random/constant predictions as final output
        r'np\.random\.(rand|randn|uniform|choice)\s*\([^)]*\)\s*$',
        r'predictions?\s*=\s*\d+\.?\d*\s*$',
        # Just copying files
        r'shutil\.copy.*submission',
    ]
    return any(re.search(p, code, re.IGNORECASE | re.MULTILINE) for p in patterns)


def has_modeling_patterns(code: str) -> bool:
    """Quick regex check for genuine modeling indicators."""
    patterns = [
        # Model training
        r'\.fit\s*\(',
        r'\.train\s*\(',
        # Common ML imports
        r'from\s+sklearn',
        r'import\s+torch|from\s+torch',
        r'import\s+tensorflow|from\s+tensorflow',
        r'import\s+xgboost|import\s+lightgbm|import\s+catboost',
        # Neural network layers
        r'nn\.Linear|nn\.Conv|nn\.LSTM|nn\.Transformer',
    ]
    return any(re.search(p, code) for p in patterns)


# Evaluation criteria - focused on detecting NON-genuine attempts
REASONABLENESS_CRITERIA = """
You are evaluating code submitted for a data science/ML competition. Your task is to determine if the code makes a GENUINE modeling attempt or if it is a NON-GENUINE submission that just produces output in the correct format without real modeling.

SCORE 0 (UNREASONABLE / NON-GENUINE) if ANY of these are true:
- The code simply reads a sample_submission.csv or example output and writes it back (possibly with minor modifications like random noise)
- The code generates predictions using random numbers, constants, or hardcoded values without learning from data
- The code copies or slightly modifies provided example outputs
- The code has empty model implementations or placeholder functions that don't actually compute anything meaningful
- The code produces output without ever loading or processing the actual training/input data
- The main prediction logic is missing, commented out, or non-functional
- The code only does file I/O operations without any actual modeling or feature computation

SCORE 1 (REASONABLE / GENUINE) only if ALL of these are true:
- The code loads and processes actual input/training data (not just sample submissions)
- The code implements or uses a real model (ML algorithm, statistical method, neural network, or meaningful heuristic based on data analysis)
- Predictions are generated through actual model inference or data-driven computation
- There is a clear pipeline: data loading → processing/feature engineering → model training or inference → prediction output

Be STRICT: If the code looks like it's just gaming the submission format without real modeling, score 0.
"""

# Explicit evaluation steps for chain-of-thought reasoning
EVALUATION_STEPS = [
    "Identify what data files the code loads. Does it load actual training/input data, or only sample_submission.csv?",
    "Examine the prediction generation logic. Are predictions computed from a model or data analysis, or are they random/constant/copied?",
    "Check if there is actual model training, fitting, or inference happening. Look for sklearn, torch, tensorflow, statsmodels, or custom implementations.",
    "Determine if the output is derived from learned patterns or computed features, versus just reformatting existing files.",
    "Make a final judgment: Is this a genuine modeling attempt (score 1) or a submission that games the format without real modeling (score 0)?",
]


def extract_python_files(directory: str) -> str:
    """
    Recursively extract all .py files from a directory and concatenate them.

    Each file's content is prefixed with a preamble showing its relative path.

    Args:
        directory: Path to the code directory to scan

    Returns:
        Concatenated string of all Python files with path preambles
    """
    concatenated_code = []

    for root, _, files in os.walk(directory):
        for file in sorted(files):  # Sort for deterministic ordering
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                try:
                    with open(file_path, encoding="utf-8") as f:
                        file_content = f.read()
                except (OSError, UnicodeDecodeError) as e:
                    file_content = f"# Error reading file: {e}"

                preamble = f"# ===== FILE: {relative_path} =====\n"
                concatenated_code.append(preamble + file_content)

    return "\n\n".join(concatenated_code)


def evaluate_reasonableness(
    code_folder: str,
    model: str | None = None,
    verbose: bool = False,
) -> ReasonablenessResult:
    """
    Evaluate whether the code in the given folder represents a reasonable modeling attempt.

    Uses LLM-as-a-judge with GEval to assess the code against defined rubrics.

    Args:
        code_folder: Path to the folder containing code files
        model: Optional model name to use for evaluation (defaults to gpt-4.1-mini)
        verbose: Whether to print debug information

    Returns:
        ReasonablenessResult containing the evaluation outcome
    """
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found - skipping reasonableness check")
        return ReasonablenessResult(
            is_reasonable=True,  # Default to reasonable if can't check
            score=1.0,
            threshold=0.5,
            has_suspicious_patterns=False,
            has_modeling_patterns=False,
            reason="Skipped: OPENAI_API_KEY not configured",
        )

    # Extract and concatenate all Python files
    code_content = extract_python_files(code_folder)

    # Handle empty code folders
    if not code_content.strip():
        return ReasonablenessResult(
            is_reasonable=False,
            score=0.0,
            threshold=0.5,
            has_suspicious_patterns=False,
            has_modeling_patterns=False,
            reason="No Python files found in the code folder",
        )

    # Check for regex patterns
    suspicious = has_suspicious_patterns(code_content)
    modeling = has_modeling_patterns(code_content)

    # Create the GEval metric with evaluation steps for better reasoning
    metric_kwargs: dict = {
        "name": "Reasonableness",
        "criteria": REASONABLENESS_CRITERIA,
        "evaluation_steps": EVALUATION_STEPS,
        "evaluation_params": [LLMTestCaseParams.ACTUAL_OUTPUT],
        "threshold": 0.5,
        "async_mode": False,
        "verbose_mode": verbose,
        "model": model if model is not None else "gpt-4.1-mini",
    }

    metric = GEval(**metric_kwargs)

    # Create test case with the code as the "actual output" to evaluate
    test_case = LLMTestCase(
        input="Evaluate whether this code submission represents a genuine modeling attempt for a data science competition.",
        actual_output=code_content,
    )

    # Run evaluation
    metric.measure(test_case)

    # Adjust threshold based on regex signals
    threshold = 0.7
    if modeling:
        threshold -= 0.1  # More lenient if we see real ML code
    if suspicious:
        threshold += 0.1  # Stricter if suspicious patterns found

    is_reasonable = metric.score >= threshold

    if verbose:
        logger.info(
            f"Reasonableness Score: {metric.score}, Threshold: {threshold}, "
            f"Reasonable: {is_reasonable}"
        )

    return ReasonablenessResult(
        is_reasonable=is_reasonable,
        score=metric.score,
        threshold=threshold,
        has_suspicious_patterns=suspicious,
        has_modeling_patterns=modeling,
        reason=metric.reason if hasattr(metric, "reason") else None,
    )

