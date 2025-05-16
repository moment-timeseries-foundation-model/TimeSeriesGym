import argparse
import json
from pathlib import Path

from timeseriesgym.data import (
    download_and_prepare_dataset,
    ensure_leaderboard_exists,
    fully_cleanup_competition,
    remove_zip_files,
)
from timeseriesgym.grade import (
    grade_code,
    grade_hyperparameter_search,
    grade_jsonl,
    grade_sample,
)
from timeseriesgym.judge import llm_judge
from timeseriesgym.registry import registry
from timeseriesgym.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Runs agents on Kaggle competitions.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run.")

    # Prepare sub-parser
    parser_prepare = subparsers.add_parser(
        name="prepare",
        help="Download and prepare competitions for the TimeSeriesGym dataset.",
    )
    parser_prepare.add_argument(
        "-c",
        "--competition-id",
        help=f"ID of the competition to prepare. Valid options: {registry.list_competition_ids()}",
        type=str,
        required=False,
    )
    parser_prepare.add_argument(
        "-a",
        "--all",
        help="Prepare all competitions.",
        action="store_true",
    )
    parser_prepare.add_argument(
        "--lite",
        help="Prepare all the low complexity competitions (TimeSeriesGym Lite).",
        action="store_true",
        required=False,
    )
    parser_prepare.add_argument(
        "-l",
        "--list",
        help="Prepare a list of competitions specified line by line in a text file.",
        type=str,
        required=False,
    )
    parser_prepare.add_argument(
        "--keep-raw",
        help="Keep the raw competition files after the competition has been prepared.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default=registry.get_data_dir(),
    )
    parser_prepare.add_argument(
        "--overwrite-checksums",
        help="[For Developers] Overwrite the checksums file for the competition.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--overwrite-leaderboard",
        help="[For Developers] Overwrite the leaderboard file for the competition.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--skip-verification",
        help="[For Developers] Skip the verification of the checksums.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--skip-leaderboard",
        help="[For Developers] Skip downloading & verifying the leaderboard.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--allow-empty-private-dir",
        help="[For Developers] Allow empty private directories in the competition data.",
        action="store_true",
        required=False,
        default=False,
    )
    parser_prepare.add_argument(
        "--allow-empty-answers-file",
        help="[For Developers] Allow empty answers files in the competition data.",
        action="store_true",
        required=False,
        default=False,
    )

    # Grade eval sub-parser
    parser_grade_eval = subparsers.add_parser(
        "grade",
        help="Grade a submission to the eval, comprising of several competition submissions",
    )
    parser_grade_eval.add_argument(
        "--submission",
        help="Path to the JSONL file of submissions. Refer to README.md#submission-format for the "
        "required format.",
        type=str,
        required=True,
    )
    parser_grade_eval.add_argument(
        "--output-dir",
        help="Path to the directory where the evaluation metrics will be saved.",
        type=str,
        required=True,
    )
    parser_grade_eval.add_argument(
        "--data-dir",
        help="Path to the directory where the data used for grading is stored.",
        required=False,
        default=registry.get_data_dir(),
    )

    # Grade sample sub-parser
    parser_grade_sample = subparsers.add_parser(
        name="grade-sample",
        help="Grade a single sample (competition) in the eval",
    )
    parser_grade_sample.add_argument(
        "submission",
        help="Path to the submission CSV file.",
        type=str,
    )
    parser_grade_sample.add_argument(
        "competition_id",
        help=f"ID of the competition to grade. Valid options: {registry.list_competition_ids()}",
        type=str,
    )
    parser_grade_sample.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default=registry.get_data_dir(),
    )

    # Grade model sub-parser
    parser_grade_model = subparsers.add_parser(
        name="grade-hyperparameter-search",
        help="Grade multiple files in a given submission folder.",
    )
    parser_grade_model.add_argument(
        "folder_path",
        help="Path to the folder containing the submission files.",
        type=str,
    )
    parser_grade_model.add_argument(
        "competition_id",
        help=f"ID of the competition to grade. Valid options: {registry.list_competition_ids()}",
        type=str,
    )
    parser_grade_model.add_argument(
        "--solution-file-path",
        help="Path to the solution script file.",
        type=str,
        required=False,
        default=None,
    )
    parser_grade_model.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default=registry.get_data_dir(),
    )

    # Grade code sub-parser
    parser_grade_code = subparsers.add_parser(
        name="grade-code",
        help="Grade a single code submission file.",
    )
    parser_grade_code.add_argument(
        "submission",
        help="Path to the submission .py file.",
        type=str,
    )
    parser_grade_code.add_argument(
        "competition_id",
        help=f"ID of the competition to grade. Valid options: {registry.list_competition_ids()}",
        type=str,
    )
    parser_grade_code.add_argument(
        "--data-dir",
        help="Path to the directory where the data will be stored.",
        required=False,
        default=registry.get_data_dir(),
    )

    # Dev tools sub-parser
    parser_dev = subparsers.add_parser("dev", help="Developer tools for extending TimeSeriesGym.")
    dev_subparsers = parser_dev.add_subparsers(dest="dev_command", help="Developer command to run.")

    # Set up 'download-leaderboard' under 'dev'
    parser_download_leaderboard = dev_subparsers.add_parser(
        "download-leaderboard",
        help="Download the leaderboard for a competition.",
    )
    parser_download_leaderboard.add_argument(
        "-c",
        "--competition-id",
        help=(
            "Name of the competition to download the leaderboard for. Valid options: "
            f"{registry.list_competition_ids()}"
        ),
        type=str,
        required=False,
    )
    parser_download_leaderboard.add_argument(
        "--all",
        help="Download the leaderboard for all competitions.",
        action="store_true",
    )
    parser_download_leaderboard.add_argument(
        "--force",
        help="Force download the leaderboard, even if it already exists.",
        action="store_true",
    )

    # Cleanup sub-parser
    parser_cleanup = subparsers.add_parser(
        "cleanup",
        help="Clean up competition files to free up disk space.",
    )
    parser_cleanup.add_argument(
        "-c",
        "--competition-id",
        help=f"ID of the competition to clean up. Valid options: {registry.list_competition_ids()}",
        type=str,
        required=False,
    )
    parser_cleanup.add_argument(
        "-a",
        "--all",
        help="Clean up all competitions.",
        action="store_true",
    )
    parser_cleanup.add_argument(
        "--data-dir",
        help="Path to the directory where the data is stored.",
        required=False,
        default=registry.get_data_dir(),
    )
    parser_cleanup.add_argument(
        "--full",
        help="Perform a full cleanup by removing the entire competition directory.",
        action="store_true",
    )
    parser_cleanup.add_argument(
        "--zips-only",
        help="Remove only the competition zip files, keeping other files.",
        action="store_true",
        default=True,
    )
    parser_cleanup.add_argument(
        "-y",
        "--yes",
        help="Automatically answer yes to all prompts.",
        action="store_true",
    )

    parser_judge = subparsers.add_parser(
        "judge",
        help="Run LLM-as-a-judge evaluation using a config.yaml specifying artifacts and criteria.",
    )
    parser_judge.add_argument(
        "config",
        help="Path to the config.yaml file specifying files and evaluation criteria.",
        type=str,
    )
    parser_judge.add_argument(
        "--model",
        help="OpenAI model to use for evaluation (default: gpt-4o)",
        type=str,
        default="gpt-4o",
    )
    parser_judge.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (silent mode)",
        default=False,
    )

    args = parser.parse_args()

    if args.command == "prepare":
        new_registry = registry.set_data_dir(Path(args.data_dir))

        if args.lite:
            competitions = [
                new_registry.get_competition(competition_id)
                for competition_id in new_registry.get_lite_competition_ids()
            ]
        elif args.all:
            competitions = [
                new_registry.get_competition(competition_id)
                for competition_id in registry.list_competition_ids()
            ]
        elif args.list:
            with open(args.list) as f:
                competition_ids = f.read().splitlines()
            competitions = [
                new_registry.get_competition(competition_id) for competition_id in competition_ids
            ]
        else:
            if not args.competition_id:
                parser_prepare.error(
                    "One of --lite, --all, --list, or --competition-id must be specified."
                )
            competitions = [new_registry.get_competition(args.competition_id)]

        for competition in competitions:
            download_and_prepare_dataset(
                competition=competition,
                keep_raw=args.keep_raw,
                overwrite_checksums=args.overwrite_checksums,
                overwrite_leaderboard=args.overwrite_leaderboard,
                skip_verification=args.skip_verification,
                skip_leaderboard=args.skip_leaderboard,
                allow_empty_private_dir=args.allow_empty_private_dir,
                allow_empty_answers_file=args.allow_empty_answers_file,
            )
    if args.command == "grade":
        new_registry = registry.set_data_dir(Path(args.data_dir))
        submission = Path(args.submission)
        output_dir = Path(args.output_dir)
        grade_jsonl(submission, output_dir, new_registry)
    if args.command == "grade-sample":
        new_registry = registry.set_data_dir(Path(args.data_dir))
        competition = new_registry.get_competition(args.competition_id)
        submission = Path(args.submission)

        report = grade_sample(submission, competition)
        logger.info("Competition report:")
        logger.info(json.dumps(report.to_dict(), indent=4))
    if args.command == "grade-hyperparameter-search":
        new_registry = registry.set_data_dir(Path(args.data_dir))
        competition = new_registry.get_competition(args.competition_id)
        submission = Path(args.folder_path)

        report = grade_hyperparameter_search(
            submission, competition, path_to_solution=args.solution_file_path
        )
        logger.info("Competition report:")
        logger.info(json.dumps(report.to_dict(), indent=4))
    if args.command == "grade-code":
        new_registry = registry.set_data_dir(Path(args.data_dir))
        competition = new_registry.get_competition(args.competition_id)
        submission = Path(args.submission)

        report = grade_code(submission, competition)
        logger.info("Competition report:")
        logger.info(json.dumps(report.to_dict(), indent=4))
    if args.command == "dev":
        if args.dev_command == "download-leaderboard":
            if args.all:
                for competition_id in registry.list_competition_ids():
                    competition = registry.get_competition(competition_id)
                    ensure_leaderboard_exists(competition, force=args.force)
            elif args.competition_id:
                competition = registry.get_competition(args.competition_id)
                ensure_leaderboard_exists(competition, force=args.force)
            else:
                parser_download_leaderboard.error(
                    "Either --all or --competition-id must be specified."
                )

    if args.command == "cleanup":
        new_registry = registry.set_data_dir(Path(args.data_dir))

        if args.all:
            if args.full:
                message = (
                    "⚠️  WARNING: This will delete ALL competition directories and their contents. ⚠️"
                )
            else:
                message = "⚠️  WARNING: This will delete ALL competition zip files. ⚠️"

            if not args.yes:
                confirm = input(f"{message}\nAre you sure you want to continue? [y/N]: ")
                if confirm.lower() != "y":
                    logger.info("Operation cancelled.")
                    return

            logger.info(
                f"{'Fully cleaning up' if args.full else 'Removing zip files for'} "
                f"all competitions..."
            )
            for competition_id in registry.list_competition_ids():
                competition = new_registry.get_competition(competition_id)
                if args.full:
                    fully_cleanup_competition(competition)
                else:
                    remove_zip_files(competition)
        elif args.competition_id:
            competition = new_registry.get_competition(args.competition_id)

            if args.full:
                message = (
                    f"⚠️  WARNING: This will delete the entire directory for competition "
                    f"`{args.competition_id}`. ⚠️"
                )
                if not args.yes:
                    confirm = input(f"{message}\nAre you sure you want to continue? [y/N]: ")
                    if confirm.lower() != "y":
                        logger.info("Operation cancelled.")
                        return
                fully_cleanup_competition(competition)
            else:
                remove_zip_files(competition)
        else:
            parser_cleanup.error("Either --all or --competition-id must be specified.")

    if args.command == "judge":
        llm_judge(args.config, args.model, args.json)


if __name__ == "__main__":
    main()
