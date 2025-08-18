#!/usr/bin/env python
"""
Script to find non-Kaggle competitions and run their download_data.py scripts.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_competitions_with_download_script(base_path):
    """
    Find all competitions that have a download_data.py script.

    Args:
        base_path: Base path to search for competitions

    Returns:
        List of paths to download_data.py scripts
    """
    competition_path = Path(base_path) / "timeseriesgym" / "competitions"
    download_scripts = []

    if not competition_path.exists():
        print(f"Competition path not found: {competition_path}")
        return download_scripts

    # Find all download_data.py files
    for download_script in competition_path.glob("**/download_data.py"):
        download_scripts.append(download_script)

    return download_scripts


def run_download_script(script_path, data_dir=None, force=False):
    """
    Run a download_data.py script.

    Args:
        script_path: Path to the download_data.py script
        data_dir: Directory where data should be stored
        force: Whether to force download even if files exist

    Returns:
        Boolean indicating success
    """
    try:
        cmd = [sys.executable, str(script_path)]

        if data_dir:
            cmd.extend(["--data-dir", str(data_dir)])

        if force:
            cmd.append("--force")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print(f"Successfully ran {script_path.name}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error running {script_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Find and run download scripts for non-Kaggle competitions."
    )
    parser.add_argument(
        "--base-path",
        default=".",
        help="Base path of the TimeSeriesGym project (default: current directory)",
    )
    parser.add_argument("--data-dir", help="Directory where data should be stored")
    parser.add_argument(
        "--force", action="store_true", help="Force download even if files already exist"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be run without executing"
    )

    args = parser.parse_args()

    # Find all download scripts
    download_scripts = find_competitions_with_download_script(args.base_path)

    if not download_scripts:
        print("No download_data.py scripts found.")
        return

    print(f"Found {len(download_scripts)} competitions with download scripts:")
    for script in download_scripts:
        competition_name = script.parent.name
        print(f"  - {competition_name}")

    # Run each script
    successful = 0
    for script_path in download_scripts:
        if args.dry_run:
            print(f"Would run: {script_path}")
            successful += 1
        else:
            if run_download_script(script_path, args.data_dir, args.force):
                successful += 1

    print(
        f"\nCompleted: {successful}/{len(download_scripts)} scripts {'would be ' if args.dry_run else ''}run successfully."
    )


if __name__ == "__main__":
    main()
