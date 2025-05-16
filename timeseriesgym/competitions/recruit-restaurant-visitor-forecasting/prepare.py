import zipfile
from copy import deepcopy
from pathlib import Path

from timeseriesgym.utils import get_logger, read_csv

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path) -> None:
    stop_date = "2017-03-01"  # roughly 10%

    # first, for all file that ends with .zip, unzip it
    for file in raw.glob("*.zip"):
        logger.info(f"Unzipping {file}")
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(raw)
        # remove the zip file
        file.unlink()

    train_path = raw / "air_visit_data.csv"
    air_reserve = raw / "air_reserve.csv"
    hpg_reserve = raw / "hpg_reserve.csv"
    air_store_info = raw / "air_store_info.csv"
    date_info = raw / "date_info.csv"
    hpg_store_info = raw / "hpg_store_info.csv"
    store_id_relation = raw / "store_id_relation.csv"

    df = read_csv(train_path)
    df_air_reserve = read_csv(air_reserve)
    df_hpg_reserve = read_csv(hpg_reserve)

    df_air_store_info = read_csv(air_store_info)
    df_date_info = read_csv(date_info)
    df_hpg_store_info = read_csv(hpg_store_info)
    df_store_id_relation = read_csv(store_id_relation)

    # split by stop date
    train_df = df[df["visit_date"] < stop_date].reset_index(drop=True)
    test_df = df[df["visit_date"] >= stop_date].reset_index(drop=True)

    new_restaurants = []
    for store_id in test_df["air_store_id"].unique():
        if store_id not in train_df["air_store_id"].unique():
            new_restaurants.append(store_id)

    test_df = test_df[~test_df["air_store_id"].isin(new_restaurants)].reset_index(drop=True)

    test_df["id"] = test_df["air_store_id"].astype(str) + "_" + test_df["visit_date"].astype(str)
    test_df = test_df[["id", "visitors"]]

    # make sample submission
    sample_submission = deepcopy(test_df)
    sample_submission["visitors"] = 0

    df_air_reserve = df_air_reserve[df_air_reserve["visit_datetime"] < stop_date].reset_index(
        drop=True
    )
    df_hpg_reserve = df_hpg_reserve[df_hpg_reserve["visit_datetime"] < stop_date].reset_index(
        drop=True
    )

    train_df.to_csv(public / "train.csv", index=False)
    test_df.to_csv(private / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    df_air_reserve.to_csv(public / "air_reserve.csv", index=False)
    df_hpg_reserve.to_csv(public / "hpg_reserve.csv", index=False)
    df_air_store_info.to_csv(public / "air_store_info.csv", index=False)
    df_date_info.to_csv(public / "date_info.csv", index=False)
    df_hpg_store_info.to_csv(public / "hpg_store_info.csv", index=False)
    df_store_id_relation.to_csv(public / "store_id_relation.csv", index=False)
