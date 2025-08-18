import logging
import urllib.request
from pathlib import Path


def download_data(data_path: Path) -> str:
    """
    Download UCR anomaly detection datasets and extract them

    Args:
        data_path: Path to data directory
    """
    competition_name = "ucr-anomaly-detection"
    competition_path = data_path / competition_name
    competition_path.mkdir(parents=True, exist_ok=True)

    ucr_ad_url = (
        "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
    )

    logging.info(f"Downloading data from {ucr_ad_url}")
    status = urllib.request.urlretrieve(ucr_ad_url, competition_path / "ucr-anomaly-detection.zip")
    logging.info(f"Downloaded data to {status}")

    return competition_path


if __name__ == "__main__":
    logging.info(download_data(Path("/raid/mgoswami/TimeSeriesGym-data")))
