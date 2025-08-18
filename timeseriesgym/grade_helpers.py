"""Helper classes related to grading"""

import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from timeseriesgym.utils import InvalidSubmissionError, get_logger, import_fn

logger = get_logger(__name__)


@dataclass
class GradingInput:
    submission: pd.DataFrame | None = None
    answers: Any | None = None
    submission_folder_path: Path | None = None
    hyperparameter_search_config: dict | None = None
    solution_file_path: Path | None = None
    submission_file_path: Path | None = None
    coding_config: dict | None = None


class Grader:
    def __init__(self, name: str, grade_fn: str) -> None:
        self.name = name
        self.grade_fn = import_fn(grade_fn)
        assert isinstance(self.name, str), "Grader name must be a string."
        assert len(self.name) > 0, "Grader name cannot be empty."

    def is_lower_better(self, leaderboard: pd.DataFrame) -> bool:
        """
        Determines if a lower score is better based on the leaderboard.
        Returns True if lower scores are better, False otherwise.
        """
        scores = leaderboard["score"]
        top_score = scores.iloc[0]
        bottom_score = scores.iloc[-1]

        return bool(top_score < bottom_score)

    @staticmethod
    def from_dict(data: dict) -> "Grader":
        return Grader(**data)

    def __call__(self, grading_input: GradingInput) -> float | None:
        """
        Runs the grading function on a submission, returning the score rounded to 5 decimal places.
        """
        try:
            kwargs = {k: v for k, v in grading_input.__dict__.items() if v is not None}
            score = self.grade_fn(**kwargs)
        except InvalidSubmissionError as e:
            logger.warning(f"Invalid submission: {e}")
            return None
        except Exception as e:
            try:
                fpath = inspect.getfile(self.grade_fn)
                line_number = inspect.getsourcelines(self.grade_fn)[1]
                fn_info = f"{fpath}:{line_number}"
            except TypeError:
                fn_info = str(self.grade_fn)
            logger.error(f"Unexpected error during grading: {e}. Check {fn_info}")
            return None
        rounded_score = round(score, 5) if isinstance(score, float) else score
        return rounded_score

    def rank_score(self, score: float | None, leaderboard: pd.DataFrame) -> dict:
        """
        Ranks a score based on the leaderboard.
        Returns a dictionary of bools with the following keys:
        - gold_medal: bool
        - silver_medal: bool
        - bronze_medal: bool
        - above_median: bool
        - gold_threshold: float
        - silver_threshold: float
        - bronze_threshold: float
        - median_threshold: float
        """
        assert "score" in leaderboard.columns, "Leaderboard must have a `score` column."

        lower_is_better = self.is_lower_better(leaderboard)

        num_teams = len(leaderboard)
        scores = leaderboard["score"]

        def get_score_at_position(position: int) -> float:
            """
            Returns the score at the given position in the leaderboard.
            Raises an IndexError if the position is out of bounds.
            """
            if position - 1 >= len(scores) or position < 1:
                raise IndexError("Position out of bounds in the leaderboard.")
            return scores.iloc[position - 1]

        def get_thresholds(num_teams: int) -> tuple[float, float, float, float]:
            """
            Returns the thresholds for medals based on kaggle.com/progression
            Returns a tuple of thresholds in the order of gold, silver, bronze, median
            """
            if 1 <= num_teams < 100:
                gold_threshold = get_score_at_position(max(1, int(num_teams * 0.1)))
                silver_threshold = get_score_at_position(max(1, int(num_teams * 0.2)))
                bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.4)))
            elif 100 <= num_teams < 250:
                gold_threshold = get_score_at_position(10)
                silver_threshold = get_score_at_position(max(1, int(num_teams * 0.2)))
                bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.4)))
            elif 250 <= num_teams < 1000:
                gold_threshold = get_score_at_position(10 + int(num_teams * 0.002))
                silver_threshold = get_score_at_position(50)
                bronze_threshold = get_score_at_position(100)
            elif num_teams >= 1000:
                gold_threshold = get_score_at_position(10 + int(num_teams * 0.002))
                silver_threshold = get_score_at_position(max(1, int(num_teams * 0.05)))
                bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.1)))
            else:
                raise ValueError("Number of teams in leaderboard must be greater than 0.")

            median_threshold = scores.median()

            return (
                float(gold_threshold),
                float(silver_threshold),
                float(bronze_threshold),
                float(median_threshold),
            )

        gold_threshold, silver_threshold, bronze_threshold, median_threshold = get_thresholds(
            num_teams
        )

        if score is None:
            return {
                "gold_medal": False,
                "silver_medal": False,
                "bronze_medal": False,
                "above_median": False,
                "gold_threshold": gold_threshold,
                "silver_threshold": silver_threshold,
                "bronze_threshold": bronze_threshold,
                "median_threshold": median_threshold,
            }

        assert isinstance(
            score, float | int
        ), f"Expected `score` to be a `float` or `int` but got a {type(score)}."

        gold_medal = score <= gold_threshold if lower_is_better else score >= gold_threshold
        silver_medal = not gold_medal and (
            score <= silver_threshold if lower_is_better else score >= silver_threshold
        )
        bronze_medal = (
            not gold_medal
            and not silver_medal
            and (score <= bronze_threshold if lower_is_better else score >= bronze_threshold)
        )
        above_median = score < median_threshold if lower_is_better else score > median_threshold

        return {
            "gold_medal": gold_medal,
            "silver_medal": silver_medal,
            "bronze_medal": bronze_medal,
            "above_median": above_median,
            "gold_threshold": gold_threshold,
            "silver_threshold": silver_threshold,
            "bronze_threshold": bronze_threshold,
            "median_threshold": median_threshold,
        }


@dataclass(frozen=True)
class CompetitionReport:
    competition_id: str
    score: float | None
    gold_threshold: float | None
    silver_threshold: float | None
    bronze_threshold: float | None
    median_threshold: float | None
    any_medal: bool | None
    gold_medal: bool | None
    silver_medal: bool | None
    bronze_medal: bool | None
    above_median: bool | None
    submission_exists: bool
    valid_submission: bool
    is_lower_better: bool
    created_at: datetime
    submission_path: str

    def to_dict(self) -> dict:
        # Convert all values to JSON-compatible types explicitly
        return {
            "competition_id": self.competition_id,
            "score": str(self.score)
            if self.score is not None
            else None,  # TODO: find a better way to support none-float output
            "gold_threshold": float(self.gold_threshold),
            "silver_threshold": float(self.silver_threshold),
            "bronze_threshold": float(self.bronze_threshold),
            "median_threshold": float(self.median_threshold),
            "any_medal": bool(self.any_medal),
            "gold_medal": bool(self.gold_medal),
            "silver_medal": bool(self.silver_medal),
            "bronze_medal": bool(self.bronze_medal),
            "above_median": bool(self.above_median),
            "submission_exists": bool(self.submission_exists),
            "valid_submission": bool(self.valid_submission),
            "is_lower_better": bool(self.is_lower_better),
            "created_at": self.created_at.isoformat(),  # Serialize datetime
            "submission_path": self.submission_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompetitionReport":
        data = data.copy()  # Avoid accidentally mutating the original dictionary
        typed_data = {
            "competition_id": data["competition_id"],
            "score": float(data["score"]) if data["score"] is not None else None,
            "gold_threshold": float(data["gold_threshold"]),
            "silver_threshold": float(data["silver_threshold"]),
            "bronze_threshold": float(data["bronze_threshold"]),
            "median_threshold": float(data["median_threshold"]),
            "any_medal": bool(data["any_medal"]),
            "gold_medal": bool(data["gold_medal"]),
            "silver_medal": bool(data["silver_medal"]),
            "bronze_medal": bool(data["bronze_medal"]),
            "above_median": bool(data["above_median"]),
            "submission_exists": bool(data["submission_exists"]),
            "valid_submission": bool(data["valid_submission"]),
            "is_lower_better": bool(data["is_lower_better"]),
            "created_at": datetime.fromisoformat(data["created_at"]),
            "submission_path": data["submission_path"],
        }

        return cls(**typed_data)


@dataclass(frozen=True)
class CodeCompetitionReport:
    competition_id: str
    defined_classes: str | None
    initialized_classes: str | None
    defined_class_methods: str | None
    executed_class_methods: str | None
    defined_functions: str | None
    executed_functions: str | None
    test_metric: str | None
    submission_exists: bool
    valid_submission: bool
    created_at: datetime
    submission_path: str

    def to_dict(self) -> dict:
        # Convert all values to JSON-compatible types explicitly
        return {
            "competition_id": self.competition_id,
            "defined_classes": self.defined_classes,
            "initialized_classes": self.initialized_classes,
            "defined_class_methods": self.defined_class_methods,
            "executed_class_methods": self.executed_class_methods,
            "defined_functions": self.defined_functions,
            "executed_functions": self.executed_functions,
            "test_metric": self.test_metric,
            "submission_exists": bool(self.submission_exists),
            "valid_submission": bool(self.valid_submission),
            "created_at": self.created_at.isoformat(),  # Serialize datetime
            "submission_path": self.submission_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeCompetitionReport":
        data = data.copy()  # Avoid accidentally mutating the original dictionary
        typed_data = {
            "competition_id": data["competition_id"],
            "defined_classes": data["defined_classes"],
            "initialized_classes": data["initialized_classes"],
            "defined_class_methods": data["defined_class_methods"],
            "executed_class_methods": data["executed_class_methods"],
            "defined_functions": data["defined_functions"],
            "executed_functions": data["executed_functions"],
            "test_metric": data["test_metric"],
            "submission_exists": bool(data["submission_exists"]),
            "valid_submission": bool(data["valid_submission"]),
            "created_at": datetime.fromisoformat(data["created_at"]),
            "submission_path": data["submission_path"],
        }

        return cls(**typed_data)
