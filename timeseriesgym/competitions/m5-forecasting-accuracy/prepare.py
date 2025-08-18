import json
from pathlib import Path

from timeseriesgym.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    entire_dataset = read_csv(raw / "sales_train_evaluation.csv")
    calendar = read_csv(raw / "calendar.csv")
    prices = read_csv(raw / "sell_prices.csv")

    train_dates = calendar[calendar["d"] <= "d_1913"]
    train_prices = prices[prices["wm_yr_wk"].isin(train_dates["wm_yr_wk"].unique())]
    sample_submission = read_csv(raw / "sample_submission.csv")

    # since we have just one evaluation period (private set unavaliable), we drop sample submission
    # rows whose id contains evaluation
    sample_submission = sample_submission[~sample_submission["id"].str.contains("evaluation")]

    # item related columns, we will need to keep this for both train and test set
    item_related_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # train set: d1 - d1913
    train_days = [f"d_{i}" for i in range(1, 1914)]
    train = entire_dataset[item_related_columns + train_days]

    # move to public set
    train.to_csv(public / "train.csv", index=False)
    calendar.to_csv(public / "calendar.csv", index=False)
    train_prices.to_csv(public / "sell_prices.csv", index=False)

    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # timeseriesgym expects one answer file, we will have to merge sell_prices, calendar and entire
    # dataset into one json file
    output_file = private / "test.jsonl"

    with open(output_file, "w", encoding="utf-8") as fout:
        # Process entire_dataset
        for record in entire_dataset.to_dict(orient="records"):
            record["source"] = "test"
            fout.write(json.dumps(record) + "\n")
        # Process calendar
        for record in calendar.to_dict(orient="records"):
            record["source"] = "calendar"
            fout.write(json.dumps(record) + "\n")
        # Process sell_prices
        for record in prices.to_dict(orient="records"):
            record["source"] = "sell_prices"
            fout.write(json.dumps(record) + "\n")
        # process sample_submission
        for record in sample_submission.to_dict(orient="records"):
            record["source"] = "sample_submission"
            fout.write(json.dumps(record) + "\n")
