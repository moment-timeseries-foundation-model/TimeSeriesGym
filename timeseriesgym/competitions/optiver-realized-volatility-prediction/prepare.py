import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_stock_id(s: str) -> str:
    return int(s.split("=")[-1])


def prepare(raw: Path, public: Path, private: Path):
    random_state = 0  # Used for reproducibility
    test_size = 0.1  # 10% of the data will be used for testing

    old_train_ids = sorted(
        [folder.name for folder in (raw / "book_train.parquet").glob("*") if folder.is_dir()]
    )
    stock_ids = sorted({get_stock_id(s) for s in old_train_ids})
    new_train_stock_ids, new_test_stock_ids = train_test_split(
        stock_ids, test_size=test_size, random_state=random_state
    )

    assert (
        len(new_train_stock_ids) >= 1
    ), "Expected the new train set to have at least one stock id. Got 0 stock id."

    assert (
        len(new_test_stock_ids) >= 1
    ), "Expected the new test set to have at least one stock id. Got 0 stock id."

    new_train_ids = sorted([i for i in old_train_ids if get_stock_id(i) in new_train_stock_ids])
    new_test_ids = sorted([i for i in old_train_ids if get_stock_id(i) in new_test_stock_ids])

    assert len(set(new_train_ids).intersection(set(new_test_ids))) == 0, (
        f"Expected the new train and test instances to be disjoint. Got an intersection of "
        f"{set(new_train_ids).intersection(set(new_test_ids))}."
    )

    assert len(new_train_ids) + len(new_test_ids) == len(old_train_ids), (
        f"Expected the number of new train and test instances to sum up to the number of old train "
        f"instances. Got {len(new_train_ids)} new train instances and {len(new_test_ids)} new test "
        f"instances which sum to {len(new_train_ids) + len(new_test_ids)} instead of "
        f"{len(old_train_ids)}."
    )

    assert set(new_train_ids).intersection(new_test_ids) == set(), (
        f"Expected the new train and test instances to be disjoint. Got an intersection of "
        f"{set(new_train_ids).intersection(new_test_ids)}."
    )

    for new_train_id in new_train_ids:
        shutil.copytree(
            src=raw / "book_train.parquet" / new_train_id,
            dst=public / "book_train.parquet" / new_train_id,
        )
        shutil.copytree(
            src=raw / "trade_train.parquet" / new_train_id,
            dst=public / "trade_train.parquet" / new_train_id,
        )

    for new_test_id in new_test_ids:
        shutil.copytree(
            src=raw / "book_train.parquet" / new_test_id,
            dst=public / "book_test.parquet" / new_test_id,
        )
        shutil.copytree(
            src=raw / "trade_train.parquet" / new_test_id,
            dst=public / "trade_test.parquet" / new_test_id,
        )

    # Construct new train and test ground truth csvs
    raw_df = pd.read_csv(raw / "train.csv")
    train_df = raw_df.loc[raw_df["stock_id"].isin(new_train_stock_ids)]
    train_df.to_csv(public / "train.csv", index=False)

    test_df = raw_df.loc[raw_df["stock_id"].isin(new_test_stock_ids)]
    test_df["row_id"] = test_df["stock_id"].astype(str) + "-" + test_df["time_id"].astype(str)
    # provide the mapping between the other data files and the submission file
    # provide the mapping of all test examples, instead of only a few rows as in the test.csv in
    # description.md
    test_df[["stock_id", "time_id", "row_id"]].to_csv(public / "test.csv", index=False)

    test_df = test_df[["row_id", "target"]]
    test_df.to_csv(private / "test.csv", index=False)

    actual_stock_ids = {row_id.split("-")[0] for row_id in test_df["row_id"]}

    assert len(actual_stock_ids) == len(new_test_ids), (
        f"Expected the new test instances to have {len(new_test_ids)} unique stock IDs. Got "
        f"{len(test_df['stock_id'].unique())} unique stock IDs."
    )

    sample_submission = test_df.copy()
    sample_submission.loc[:, "target"] = 0.0

    assert len(sample_submission) == len(test_df), (
        f"Expected the sample submission to have the same number of instances as the new test "
        f"instances. Got {len(sample_submission)} instances in the sample submission and "
        f"{len(test_df)} new test instances."
    )

    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    assert sorted(public.glob("book_train.parquet/*")) == sorted(
        {public / "book_train.parquet" / stock_id for stock_id in new_train_ids}
    ), "Expected the public book_train.parquet directory to contain the new train instances."

    assert sorted(public.glob("trade_train.parquet/*")) == sorted(
        {public / "trade_train.parquet" / stock_id for stock_id in new_train_ids}
    ), "Expected the public trade_train.parquet directory to contain the new train instances."

    assert sorted(public.glob("book_test.parquet/*")) == sorted(
        {public / "book_test.parquet" / stock_id for stock_id in new_test_ids}
    ), "Expected the public book_test.parquet directory to contain the new test instances."

    assert sorted(public.glob("trade_test.parquet/*")) == sorted(
        {public / "trade_test.parquet" / stock_id for stock_id in new_test_ids}
    ), "Expected the public trade_test.parquet directory to contain the new test instances."

    assert (
        len(list(public.rglob("train.csv"))) == 1
    ), "Expected the public directory to contain only one train ground truth file."
