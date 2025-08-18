"""High-level grading functionality"""

import json
from datetime import datetime
from pathlib import Path
import traceback

from tqdm import tqdm

from timeseriesgym.data import get_leaderboard, is_dataset_prepared
from timeseriesgym.grade_helpers import CodeCompetitionReport, CompetitionReport, GradingInput
from timeseriesgym.registry import Competition, Registry
from timeseriesgym.registry import registry as default_registry
from timeseriesgym.utils import (
    InvalidSubmissionError,
    get_logger,
    get_null_rank_info,
    get_timestamp,
    load_answers,
    load_submissions,
    purple,
    read_jsonl,
)

logger = get_logger(__name__)


def grade_jsonl(
    path_to_submissions: Path,
    output_dir: Path,
    registry: Registry = default_registry,
):
    """
    Grades multiple submissions stored in a JSONL file.
    Saves the aggregated report as a JSON file.
    """

    submissions = read_jsonl(path_to_submissions, skip_commented_out_lines=True)
    competitions_reports = []

    for submission in tqdm(submissions, desc="Grading submissions", unit="submission"):
        submission_path = Path(str(submission["submission_path"]))
        competition_id = submission["competition_id"]
        competition = registry.get_competition(competition_id)
        single_report = grade_sample(submission_path, competition)
        competitions_reports.append(single_report)

    aggregated_report = aggregate_reports(competitions_reports)
    timestamp = get_timestamp()
    save_path = output_dir / f"{timestamp}_grading_report.json"
    logger.info(
        json.dumps(
            {k: v for k, v in aggregated_report.items() if k != "competition_reports"}, indent=4
        )
    )

    output_dir.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(aggregated_report, f, indent=2)
    logger.info(purple(f"Saved summary report to {save_path}"))


def grade_sample(path_to_submission: Path, competition: Competition) -> CompetitionReport:
    """Grades a submission for the given competition."""

    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
            f"Please run `timeseriesgym prepare -c {competition.id}` to prepare the dataset."
        )

    if competition.sample_submission.suffix != path_to_submission.suffix:
        raise InvalidSubmissionError(
            f"Submission invalid! Submission file {path_to_submission.suffix} is different from "
            f"the expected file extension {competition.sample_submission.suffix}"
        )

    score = None
    submission_exists = path_to_submission.is_file()
    submission_file = load_submissions(path_to_submission)
    answers = load_answers(competition.answers)
    grading_input = GradingInput(submission=submission_file, answers=answers)
    score = competition.grader(grading_input)

    valid_submission = score is not None
    competition_leaderboard = get_leaderboard(competition)

    if competition_leaderboard is not None:
        rank_info = competition.grader.rank_score(score, competition_leaderboard)
        is_lower_better = competition.grader.is_lower_better(competition_leaderboard)
    else:
        rank_info = get_null_rank_info()
        is_lower_better = None

    return CompetitionReport(
        competition_id=competition.id,
        score=score,
        gold_threshold=rank_info["gold_threshold"],
        silver_threshold=rank_info["silver_threshold"],
        bronze_threshold=rank_info["bronze_threshold"],
        median_threshold=rank_info["median_threshold"],
        any_medal=rank_info["gold_medal"] or rank_info["silver_medal"] or rank_info["bronze_medal"],
        gold_medal=rank_info["gold_medal"],
        silver_medal=rank_info["silver_medal"],
        bronze_medal=rank_info["bronze_medal"],
        above_median=rank_info["above_median"],
        submission_exists=submission_exists,
        valid_submission=valid_submission,
        is_lower_better=is_lower_better,
        created_at=datetime.now(),
        submission_path=str(path_to_submission),
    )


def grade_hyperparameter_search(
    path_to_submission_folder: Path,
    competition: Competition,
    path_to_solution: str | None = None,
) -> CompetitionReport:
    """
    Path to submission folder should contain multiple required submission files
    specified in description
    competition: Competition object
    """
    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
            f"Please run `timeseriesgym prepare -c {competition.id}` to prepare the dataset."
        )

    score = None
    answers = load_answers(competition.answers)
    hyperparameter_search_config = competition.hyperparameter_search_config

    if hyperparameter_search_config is None:
        raise ValueError(
            f"Hyperparameter searching config is not defined for competition `{competition.id}`! "
            f"Please define the hyperparameter searching config in the competition config file"
        )

    grading_input = GradingInput(
        submission_folder_path=path_to_submission_folder,
        answers=answers,
        hyperparameter_search_config=hyperparameter_search_config,
        solution_file_path=Path(path_to_solution) if path_to_solution else None,
    )
    score = competition.grader(grading_input)

    submission_exists = path_to_submission_folder.is_dir()
    valid_submission = score is not None
    competition_leaderboard = get_leaderboard(competition)

    if competition_leaderboard is not None:
        rank_info = competition.grader.rank_score(score, competition_leaderboard)
        is_lower_better = competition.grader.is_lower_better(competition_leaderboard)
    else:
        rank_info = get_null_rank_info()
        is_lower_better = None

    return CompetitionReport(
        competition_id=competition.id,
        score=score,
        gold_threshold=rank_info["gold_threshold"],
        silver_threshold=rank_info["silver_threshold"],
        bronze_threshold=rank_info["bronze_threshold"],
        median_threshold=rank_info["median_threshold"],
        any_medal=rank_info["gold_medal"] or rank_info["silver_medal"] or rank_info["bronze_medal"],
        gold_medal=rank_info["gold_medal"],
        silver_medal=rank_info["silver_medal"],
        bronze_medal=rank_info["bronze_medal"],
        above_median=rank_info["above_median"],
        submission_exists=submission_exists,
        valid_submission=valid_submission,
        is_lower_better=is_lower_better,
        created_at=datetime.now(),
        submission_path=str(path_to_submission_folder),
    )


def grade_code(path_to_submission: Path, competition: Competition) -> CompetitionReport:
    """Grades a code submission for the given competition."""

    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
            f"Please run `timeseriesgym prepare -c {competition.id}` to prepare the dataset."
        )

    if competition.sample_submission.suffix != path_to_submission.suffix:
        raise InvalidSubmissionError(
            f"Submission invalid! Submission file {path_to_submission.suffix} is different from "
            f"the expected file extension {competition.sample_submission.suffix}"
        )

    scores = None
    submission_exists = path_to_submission.is_file()
    answers = load_answers(competition.answers)
    coding_config = competition.coding_config

    if coding_config is None:
        raise ValueError(
            f"Coding config is not defined for competition `{competition.id}`! "
            f"Please define the coding config in the competition config file"
        )

    grading_input = GradingInput(
        submission_file_path=path_to_submission,
        answers=answers,
        coding_config=coding_config,
    )
    scores = competition.grader(grading_input)

    valid_submission = scores is not None
    return CodeCompetitionReport(
        competition_id=competition.id,
        defined_classes=scores["defined_classes"],
        initialized_classes=scores["initialized_classes"],
        defined_class_methods=scores["defined_class_methods"],
        executed_class_methods=scores["executed_class_methods"],
        defined_functions=scores["defined_functions"],
        executed_functions=scores["executed_functions"],
        test_metric=scores["test_metric"],
        submission_exists=submission_exists,
        valid_submission=valid_submission,
        created_at=datetime.now(),
        submission_path=str(path_to_submission),
    )


def validate_submission(submission: Path, competition: Competition) -> tuple[bool, str]:
    """
    Validates a submission for the given competition by actually running the competition grader.
    This is designed for end users, not developers (we assume that the competition grader is
    correctly implemented and use that for validating the submission, not the other way around).
    """
    if not submission.is_file():
        return False, f"Submission invalid! Submission file {submission} does not exist."

    if competition.sample_submission.suffix != submission.suffix:
        return (
            False,
            f"Submission invalid! Submission file {submission} does not have the correct file "
            f"extension.",
        )

    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
            f"Please run `timeseriesgym prepare -c {competition.id}` to prepare the dataset."
        )

    try:
        if competition.hyperparameter_search_config is not None and submission.is_dir():
            grading_input = GradingInput(
                submission_folder_path=submission,
                answers=load_answers(competition.answers),
                hyperparameter_search_config=competition.hyperparameter_search_config,
            )
        elif competition.coding_config is not None and submission.is_file():
            grading_input = GradingInput(
                submission_file_path=submission,
                answers=load_answers(competition.answers),
                coding_config=competition.coding_config,
            )
        elif submission.is_file():
            grading_input = GradingInput(
                submission_file=load_submissions(submission),
                answers=load_answers(competition.answers),
            )
        else:
            return (
                False,
                "Submission invalid! If the submission is a directory "
                "only for hyperparameter search tasks",
            )
        competition.grader(grading_input)
    except Exception as e:
        return (
            False,
            f"Submission invalid! The attempt to grade the submission has resulted in the "
            f"following error message:\n{e}",
        )

    return True, "Submission is valid."


def aggregate_reports(competition_reports: list[CompetitionReport]) -> dict:
    """
    Builds the summary report from a list of competition reports.
    If pass_at_n is True, then aggregate performence of competitions by selecting the best
    performance per competition, otherwise sum metrics
    """

    total_gold_medals = sum(
        report.gold_medal for report in competition_reports if isinstance(report, CompetitionReport)
    )
    total_silver_medals = sum(
        report.silver_medal
        for report in competition_reports
        if isinstance(report, CompetitionReport)
    )
    total_bronze_medals = sum(
        report.bronze_medal
        for report in competition_reports
        if isinstance(report, CompetitionReport)
    )
    total_above_median = sum(
        report.above_median
        for report in competition_reports
        if isinstance(report, CompetitionReport)
    )
    total_submissions = sum(report.submission_exists for report in competition_reports)
    total_valid_submissions = sum(report.valid_submission for report in competition_reports)

    summary_report = {
        "total_runs": len(competition_reports),
        "total_runs_with_submissions": int(total_submissions),
        "total_valid_submissions": int(total_valid_submissions),
        "total_medals": int(total_gold_medals + total_silver_medals + total_bronze_medals),
        "total_gold_medals": int(total_gold_medals),
        "total_silver_medals": int(total_silver_medals),
        "total_bronze_medals": int(total_bronze_medals),
        "total_above_median": int(total_above_median),
        "competition_reports": [cr.to_dict() for cr in competition_reports],
    }

    return summary_report


def grade_lite(
    path_to_submissions: Path,
    output_dir: Path,
    registry: Registry = default_registry,
):
    """
    Grades multiple submissions stored in a JSONL file for TimeSeriesGym Lite competitions only.
    Saves the aggregated report as a JSON file.
    """
    # Filter submissions to only include those for lite competitions
    all_submissions = read_jsonl(path_to_submissions, skip_commented_out_lines=True)
    lite_submissions = [
        s for s in all_submissions if s["competition_id"] in registry.get_lite_competition_ids()
    ]

    if not lite_submissions:
        logger.warning("No submissions found for TimeSeriesGym Lite competitions.")
        return

    competitions_reports = []
    for submission in tqdm(lite_submissions, desc="Grading lite submissions", unit="submission"):
        submission_path = Path(str(submission["submission_path"]))
        competition_id = submission["competition_id"]
        competition = registry.get_competition(competition_id)

        try:
            # Determine which grading function to use based on competition configuration
            if competition.hyperparameter_search_config is not None:
                # For hyperparameter search competitions
                single_report = grade_hyperparameter_search(submission_path, competition)
            elif competition.coding_config is not None:
                # For coding competitions
                single_report = grade_code(submission_path, competition)
            else:
                # For regular sample competitions
                single_report = grade_sample(submission_path, competition)

            competitions_reports.append(single_report)
        except Exception:
            logger.error(f"Error while processing {competition_id}, {traceback.format_exc()}")

    aggregated_report = aggregate_reports(competitions_reports)
    timestamp = get_timestamp()
    save_path = output_dir / f"{timestamp}_lite_grading_report.json"
    logger.info(
        json.dumps(
            {k: v for k, v in aggregated_report.items() if k != "competition_reports"}, indent=4
        )
    )

    output_dir.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(aggregated_report, f, indent=2)
    logger.info(purple(f"Saved summary report to {save_path}"))
