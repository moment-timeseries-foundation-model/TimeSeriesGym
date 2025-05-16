import functools
import hashlib
import inspect
import os
import shutil
import webbrowser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import diskcache as dc
import pandas as pd
import yaml
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from tqdm.auto import tqdm

from timeseriesgym.competitions.constants import (
    coding_competitions,
    derived_competitions,
    kaggle_original_competitions,
    non_kaggle_original_competitions,
)
from timeseriesgym.registry import Competition, registry
from timeseriesgym.utils import (
    authenticate_kaggle_api,
    extract,
    get_diff,
    get_logger,
    get_path_to_callable,
    is_empty,
    load_yaml,
)

logger = get_logger(__name__)
cache = dc.Cache("cache", size_limit=2**26)  # 64 MB


def create_prepared_dir(competition: Competition) -> None:
    competition.public_dir.mkdir(exist_ok=True, parents=True)
    competition.private_dir.mkdir(exist_ok=True, parents=True)


def download_and_prepare_dataset(
    competition: Competition,
    keep_raw: bool = True,
    overwrite_checksums: bool = False,
    overwrite_leaderboard: bool = False,
    skip_verification: bool = False,
    skip_leaderboard: bool = False,
    allow_empty_private_dir: bool = False,
    allow_empty_answers_file: bool = False,
) -> None:
    """
    Creates a `public` and `private` directory for the competition using the `prepare_fn`,
    downloading the competition's dataset zip file and extracting it into `raw` if needed.
    """

    if competition.prepare_fn is not None:
        # Check if the provided prepare function is valid.
        assert is_valid_prepare_fn(
            competition.prepare_fn
        ), "Provided `prepare_fn` doesn't take arguments `raw`, `private` and `public`!"

    if competition.id not in coding_competitions and not skip_leaderboard:
        # coding competitions do not have a leaderboard
        ensure_leaderboard_exists(competition, force=overwrite_leaderboard)

    competition_dir = competition.raw_dir.parent
    competition.raw_dir.mkdir(exist_ok=True, parents=True)
    create_prepared_dir(competition)

    if competition.id in kaggle_original_competitions:
        # Kaggle competitions can be downloaded automatically using the Kaggle API, if the
        # user has accepted the competition rules.

        logger.info(f"{competition.id} is a Kaggle competition. Downloading its datasets!")

        zipfile = download_dataset(
            competition_id=competition.id,
            download_dir=competition_dir,
            force=False,
        )
    elif competition.id in non_kaggle_original_competitions:
        logger.info(
            f"{competition.id} is not a Kaggle competition. Currently, these datasets must be "
            "manually downloaded."
        )

        zipfile = competition_dir / f"{competition.id}.zip"

        # TODO(mononito): Some of these competitions may still have automatic download scripts.
        # We can invoke them here if they exist.
        # For now, we will assume that data has been downloaded manually, and placed in the
        # competition directory.

        if not zipfile.exists():  # Check if the zip file already exists
            raise FileNotFoundError(
                f"{zipfile} not found. For non-kaggle competitions, a zip file containing "
                f"necessary files for the prepare function should be created under {zipfile}"
            )
    elif competition.id in derived_competitions:
        # Some competitions are derived from parent competitions. While they might have to be
        # prepared and processed, they do not have to be downlaoded from the internet.
        # These competitions are derived from other downloaded and prepared competitions

        logger.info(
            f"{competition.id} is a competition derived from {competition.parent_id}. Copying "
            "files from the original prepared dataset."
        )

        zipfile = None
        data_dir = competition.public_dir.parent.parent.parent  # `raw_dir` may not exist
        new_registry = registry.set_data_dir(data_dir)
        if competition.parent_id is not None:
            parent_competition = new_registry.get_competition(competition.parent_id)
        else:
            raise ValueError(
                f"Parent competition is set to `None` for `{competition.id}`. "
                "All derived competitions must have a valid `parent_id`."
            )

        # Check if the parent competition has already been prepared.
        parent_is_prepared = is_dataset_prepared(parent_competition, grading_only=True)
        derived_competition_needs_preparation = not is_dataset_prepared(competition)
        if parent_is_prepared and derived_competition_needs_preparation:
            # Copy leaderboard and prepared files from the parent competition to the child
            # competition
            logger.info(f"Copying files from {parent_competition.name} to {competition.name}.")

            copy_leaderboard(from_competition=parent_competition, to_competition=competition)
            copy_prepared_files(from_competition=parent_competition, to_competition=competition)
        else:
            # If the parent competition is not prepared, we cannot prepare the derived competition.
            if not parent_is_prepared:
                raise ValueError(
                    f"Parent competition `{parent_competition.id}` for `{competition.id}` "
                    "has not been prepared yet. Please prepare the parent competition first."
                )
            elif not derived_competition_needs_preparation:
                logger.info(
                    f"The derived competition `{competition.id}` does not require preparation "
                    "as it is already prepared."
                )
    elif competition.id in coding_competitions:
        # Coding competitions only include reference implementations in another coding language.
        logger.info(
            f"{competition.id} is a coding competition. The dataset for this competition includes "
            "only code files."
        )
        zipfile = None

    # Generate the checksum for the zip file if it exists.
    if overwrite_checksums or not skip_verification:
        logger.info(f"Generating checksum for `{zipfile}`...")
        actual_zip_checksum = get_checksum(zipfile) if zipfile else None

        if competition.checksums.is_file() and not overwrite_checksums:
            expected_checksums = load_yaml(competition.checksums)
            expected_zip_checksum = (
                expected_checksums["zip"] if "zip" in expected_checksums else None
            )

            if actual_zip_checksum != expected_zip_checksum:
                raise ValueError(
                    f"Checksum for `{zipfile}` does not match the expected checksum! "
                    f"Expected `{expected_zip_checksum}` but got `{actual_zip_checksum}`."
                )

            logger.info(f"Checksum for `{zipfile}` matches the expected checksum.")

    if zipfile and is_empty(competition.raw_dir):
        logger.info(f"Extracting `{zipfile}` to `{competition.raw_dir}`...")
        extract(zipfile, competition.raw_dir, recursive=False)
        logger.info(f"Extracted `{zipfile}` to `{competition.raw_dir}` successfully.")

    # If the competition has already been prepared, we don't need to prepare it again, unless
    # we need to overwrite checksums. In that case, the competition must have a valid preparation
    # function. In this case we will remove the existing prepared data directory and prepare the
    # dataset again. This must only be invoked for original competitions. Derived competitions are
    # generally post processed after copying data from the parent competition.
    is_prepared = is_dataset_prepared(
        competition,
        allow_empty_answers_file=allow_empty_answers_file,
        allow_empty_private_dir=allow_empty_private_dir,
    )
    needs_preparation = not is_prepared or overwrite_checksums
    has_valid_prepare_fn = is_valid_prepare_fn(competition.prepare_fn)
    original_competition = competition.id not in derived_competitions

    if needs_preparation and has_valid_prepare_fn and original_competition:
        if competition.public_dir.parent.exists() and overwrite_checksums:
            logger.info(
                f"Removing the existing prepared data directory for `{competition.id}` since "
                "`overwrite_checksums` is set to `True`..."
            )
            shutil.rmtree(competition.public_dir.parent)
            create_prepared_dir(competition)

        logger.info(
            f"Preparing the dataset using `{competition.prepare_fn.__name__}` from "
            f"`{get_path_to_callable(competition.prepare_fn)}`..."
        )

        competition.prepare_fn(
            raw=competition.raw_dir,
            public=competition.public_dir,
            private=competition.private_dir,
        )

        logger.info(f"Data for competition `{competition.id}` prepared successfully.")
    else:
        if not needs_preparation and original_competition:
            logger.info(
                f"Dataset for `{competition.id}` already prepared. Skipping preparation step."
            )
        if needs_preparation and not has_valid_prepare_fn and original_competition:
            raise ValueError(
                f"Cannot prepare dataset for `{competition.id}` as it has no valid `prepare_fn`. "
                "Please provide a valid `prepare_fn`."
            )

    # Some scenarios can be orchestrated after the dataset has been prepared.
    is_derived_competition = competition.id in derived_competitions
    derived_competition_needs_preparation = (
        is_derived_competition and derived_competition_needs_preparation
    )

    # If the competition is derived from another competition, and has a prepare function, we must
    # invoke the prepare function again. This is useful for competitions that are derived from other
    # competitions, and have a prepare function that is not the same as the parent competition's
    # prepare function. In this case, we assume that the prepare function operates over the
    # prepared public and private directories copied from the parent, and removes any unncessary
    # files.

    if derived_competition_needs_preparation and has_valid_prepare_fn:
        # If the competition is derived from another competition, and has a prepare function,
        # we need to remove the old files before copying the new ones.

        logger.info(
            f"Preparing the dataset using `{competition.prepare_fn.__name__}` from "
            f"`{get_path_to_callable(competition.prepare_fn)}`..."
        )
        competition.prepare_fn(
            raw=competition.raw_dir,
            public=competition.public_dir,
            private=competition.private_dir,
        )
    else:
        if is_derived_competition and not derived_competition_needs_preparation:
            logger.info(
                f"Skipping preparation for derived competition `{competition.id}` as it "
                "already has been prepared."
            )
        if is_derived_competition and not has_valid_prepare_fn:
            logger.info(
                f"Skipping preparation for derived competition `{competition.id}` as it "
                "does not have a valid prepare function."
            )

    if derived_competition_needs_preparation and competition.missingness_simulator:
        logger.info(
            "Simulating missingness in the dataset using "
            f"`{competition.missingness_simulator.name}` from "
            f"`{get_path_to_callable(competition.missingness_simulator)}`..."
        )

        competition.missingness_simulator(public=competition.public_dir)

    with open(competition.public_dir / "description.md", "w") as f:
        f.write(competition.description)

    # Generate checksums for the raw zip file, public and private directories.
    if overwrite_checksums or not skip_verification:
        logger.info(f"Generating checksums for files in `{competition_dir}`...")

        actual_checksums = {
            "zip": actual_zip_checksum,
            "public": generate_checksums(competition.public_dir),
            "private": generate_checksums(competition.private_dir),
        }

        if not competition.checksums.is_file() or overwrite_checksums:
            with open(competition.checksums, "w") as file:
                yaml.dump(actual_checksums, file, default_flow_style=False)

            logger.info(f"Checksums for `{competition.id}` saved to `{competition.checksums}`.")

        expected_checksums = load_yaml(competition.checksums)

        if actual_checksums != expected_checksums:
            logger.error(f"Checksums do not match for `{competition.id}`!")

            diff = get_diff(
                actual_checksums,
                expected_checksums,
                fromfile="actual_checksums",
                tofile="expected_checksums",
            )

            raise ValueError(f"Checksums do not match for `{competition.id}`!\n{diff}")

        logger.info(f"Checksums for files in `{competition_dir}` match the expected checksums.")

    if not keep_raw:
        logger.info(f"Removing the raw data directory for `{competition.id}`...")
        shutil.rmtree(competition.raw_dir)

    assert competition.public_dir.is_dir(), "Public data directory doesn't exist."
    assert competition.private_dir.is_dir(), "Private data directory doesn't exist."

    if is_empty(competition.public_dir):
        logger.warning("Public data directory is empty!")
    if is_empty(competition.private_dir):
        logger.warning("Private data directory is empty!")


def is_dataset_prepared(
    competition: Competition,
    grading_only: bool = False,
    allow_empty_private_dir: bool = False,
    allow_empty_answers_file: bool = False,
) -> bool:
    """Checks if the competition has non-empty `public` and `private` directories with the expected
    files."""

    assert isinstance(
        competition, Competition
    ), f"Expected input to be of type `Competition` but got {type(competition)}."

    public = competition.public_dir
    private = competition.private_dir

    if not grading_only:
        if not public.is_dir():
            logger.warning("Public directory does not exist.")
            return False
        if is_empty(public):
            logger.warning("Public directory is empty.")
            return False

    if not private.is_dir():
        logger.warning("Private directory does not exist.")
        return False

    if not allow_empty_private_dir and is_empty(private):
        logger.warning("Private directory is empty.")
        return False

    if not allow_empty_answers_file and not competition.answers.is_file():
        logger.warning("Answers file does not exist.")
        return False

    if (
        not allow_empty_answers_file
        and not competition.sample_submission.is_file()
        and not grading_only
    ):
        logger.warning("Sample submission file does not exist.")
        return False

    return True


def is_api_exception(exception: Exception) -> bool:
    # only import when necessary; otherwise kaggle asks for API key on import
    from kaggle.rest import ApiException

    return isinstance(exception, ApiException)


@retry(
    retry=retry_if_exception(is_api_exception),
    stop=stop_after_attempt(3),  # stop after 3 attempts
    wait=wait_fixed(5),  # wait 5 seconds between attempts
    reraise=True,
)
def download_dataset(
    competition_id: str,
    download_dir: Path,
    quiet: bool = False,
    force: bool = False,
) -> Path:
    """Downloads the competition data as a zip file using the Kaggle API and returns the path to
    the zip file."""

    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    logger.info(f"Downloading the dataset for `{competition_id}` to `{download_dir}`...")

    api = authenticate_kaggle_api()

    # only import when necessary; otherwise kaggle asks for API key on import
    from kaggle.rest import ApiException

    try:
        api.competition_download_files(
            competition=competition_id,
            path=download_dir,
            quiet=quiet,
            force=force,
        )
    except ApiException as e:
        if _need_to_accept_rules(str(e)):
            logger.warning("You must accept the competition rules before downloading the dataset.")
            _prompt_user_to_accept_rules(competition_id)
            download_dataset(competition_id, download_dir, quiet, force)
        else:
            raise e

    zip_files = list(download_dir.glob("*.zip"))

    assert (
        len(zip_files) == 1
    ), f"Expected to download a single zip file, but found {len(zip_files)} zip files."

    zip_file = zip_files[0]

    return zip_file


def _need_to_accept_rules(error_msg: str) -> bool:
    return "You must accept this competition" in error_msg


def _prompt_user_to_accept_rules(competition_id: str) -> None:
    response = input("Would you like to open the competition page in your browser now? (y/n): ")

    if response.lower() != "y":
        raise RuntimeError("You must accept the competition rules before downloading the dataset.")

    webbrowser.open(f"https://www.kaggle.com/c/{competition_id}/rules")
    input("Press Enter to continue after you have accepted the rules...")


def is_valid_prepare_fn(preparer_fn: Any) -> bool:
    """Checks if the `preparer_fn` takes three arguments: `raw`, `public` and `private`, in that
    order."""

    try:
        sig = inspect.signature(preparer_fn)
    except (TypeError, ValueError):
        return False

    actual_params = list(sig.parameters.keys())
    expected_params = ["raw", "public", "private"]

    return actual_params == expected_params


def generate_checksums(
    target_dir: Path,
    exts: list[str] | None = None,
    exclude: list[Path] | None = None,
) -> dict:
    """
    Generate checksums for the files directly under the target directory with the specified
    extensions.

    Args:
        target_dir: directory to generate checksums for.
        exts: List of file extensions to generate checksums for.
        exclude: List of file paths to exclude from checksum generation.

    Returns:
        A dictionary of form file: checksum.
    """

    if exts is None:
        exts = [
            "csv",
            "json",
            "jsonl",
            "parquet",
            "bson",
            "h5",
            "hdf5",
            "zip",
            "arrow",
            "py",
            "npy",
            "pkl",
            "pt",
            "R",
        ]

    if exclude is None:
        exclude = []

    checksums = {}

    for ext in exts:
        fpaths = target_dir.rglob(f"*.{ext}")

        for fpath in fpaths:
            if not fpath.is_file():
                continue  # skip dirs named like `my/dir.csv/`

            if fpath in exclude:
                continue

            rel_path = fpath.relative_to(target_dir)
            checksums[str(rel_path)] = get_checksum(fpath)

    return checksums


def get_last_modified(fpath: Path) -> datetime:
    """Return the last modified time of a file."""

    return datetime.fromtimestamp(fpath.stat().st_mtime)


def file_cache(fn: Callable) -> Callable:
    """A decorator that caches results of a function with a Path argument, invalidating the cache
    when the file is modified."""

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if not (len(params) == 1 and params[0].annotation is Path):
        raise NotImplementedError("Only functions with a single `Path` argument are supported.")

    # Use `functools.wraps` to preserve the function's metadata, like the name and docstring.
    # Query the cache, but with an additional `last_modified` argument in the key, which has the
    # side effect of invalidating the cache when the file is modified.
    @functools.wraps(fn)
    def wrapper(fpath: Path) -> Any:
        last_modified = get_last_modified(fpath)
        key = (fn.__name__, str(fpath), last_modified)

        if key not in cache:
            cache[key] = fn(fpath)

        return cache[key]

    return wrapper


@file_cache
def get_checksum(fpath: Path) -> str:
    """Compute MD5 checksum of a file."""

    assert fpath.is_file(), f"Expected a file at `{fpath}`, but it doesn't exist."

    hash_md5 = hashlib.md5()
    file_size = os.path.getsize(fpath)

    # only show progress bar for large files (> ~5 GB)
    show_progress = file_size > 5_000_000_000

    with open(fpath, "rb") as f:
        for chunk in tqdm(
            iter(lambda: f.read(4_096), b""),
            total=file_size // 4096,
            unit="B",
            unit_scale=True,
            disable=not show_progress,
        ):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def ensure_leaderboard_exists(competition: Competition, force: bool = False) -> Path:
    """
    Ensures the leaderboard for a given competition exists in the competition's
    directory, returning the path to it.
    If `force` is True, the leaderboard is downloaded using the Kaggle API.
    If `force` is `false`, if the leaderboard does not exist, an error is raised.
    """
    download_dir = competition.leaderboard.parent
    leaderboard_path = competition.leaderboard
    if not force:
        if leaderboard_path.exists():
            return leaderboard_path
        else:
            raise FileNotFoundError(
                f"Leaderboard not found locally for competition `{competition.id}`. Please flag "
                "this to the developers."
            )
    api = authenticate_kaggle_api()
    leaderboard = api.competition_leaderboard_view(competition=competition.id)
    if leaderboard:
        leaderboard = [row.__dict__ for row in leaderboard]
        leaderboard_df = pd.DataFrame(leaderboard)
        leaderboard_df.drop(columns=["teamNameNullable", "teamName"], inplace=True)
        leaderboard_df.to_csv(leaderboard_path, index=False)
        logger.info(
            f"Downloaded leaderboard for competition `{competition.id}` to "
            f"`{download_dir.relative_to(Path.cwd()) / 'leaderboard.csv'}`."
        )
        return leaderboard_path
    else:
        raise RuntimeError(f"Failed to download leaderboard for competition `{competition.id}`.")


def get_leaderboard(competition: Competition) -> pd.DataFrame | None:
    leaderboard_path = competition.leaderboard
    if leaderboard_path.exists():
        logger.info(f"Leaderboard found locally for competition `{competition.id}`. Loading...")
        leaderboard_df = pd.read_csv(leaderboard_path)
        return leaderboard_df
    logger.warning(f"Leaderboard not found locally for competition `{competition.id}`.")
    return None


def copy_leaderboard(
    from_competition: Competition,
    to_competition: Competition,
) -> None:
    if from_competition.leaderboard.is_file():
        shutil.copy(from_competition.leaderboard, to_competition.leaderboard)
    logger.info(f"Copying leaderboard from {from_competition.name} to {to_competition.name}.")


def copy_prepared_files(from_competition: Competition, to_competition: Competition) -> None:
    shutil.copytree(from_competition.public_dir, to_competition.public_dir, dirs_exist_ok=True)
    shutil.copytree(from_competition.private_dir, to_competition.private_dir, dirs_exist_ok=True)

    total_files_size = 0
    for dir in [to_competition.public_dir, to_competition.private_dir]:
        total_files_size += sum(f.stat().st_size for f in dir.rglob("*") if f.is_file())
    total_files_size = total_files_size / (1024**3)  # convert to GB
    logger.warning(
        f"Copying {total_files_size:.2f} GB of data files from the parent competition..."
    )


def remove_zip_files(competition: Competition) -> None:
    """Removes all zip files for a given competition.

    Args:
        competition: The competition whose zip files are to be removed.
    """
    competition_dir = competition.raw_dir.parent
    zip_files = list(competition_dir.glob("*.zip"))

    if not zip_files:
        logger.info(f"No zip files found for competition `{competition.id}`.")
        return

    for zip_file in zip_files:
        logger.info(f"Removing zip file `{zip_file}`...")
        zip_file.unlink()
        logger.info(f"Removed zip file `{zip_file}` successfully.")


def fully_cleanup_competition(competition: Competition) -> None:
    """Removes the entire competition folder and all its contents.

    Args:
        competition: The competition to be completely removed.
    """
    competition_dir = competition.raw_dir.parent

    if not competition_dir.exists():
        logger.info(f"Competition directory `{competition_dir}` does not exist.")
        return

    logger.info(f"Removing entire competition directory `{competition_dir}`...")
    shutil.rmtree(competition_dir)
    logger.info(f"Removed competition directory `{competition_dir}` successfully.")
