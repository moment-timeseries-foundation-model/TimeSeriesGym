"""Tests for the Optiver split (optiver-realized-volatility-prediction and its derived variants).

The test is the held-out stocks at a held-out set of time buckets. `train.csv` has no target at a
test time bucket, for any stock, and the book and trade files keep every stock's data at every
time bucket.
"""

import hashlib
import importlib

import numpy as np
import pandas as pd

N_STOCKS, N_TIMES = 20, 40


def _prepare():
    return importlib.import_module(
        "timeseriesgym.competitions.optiver-realized-volatility-prediction.prepare"
    ).prepare


def _raw(root):
    """Kaggle's layout: book/trade parquet partitioned by stock_id, and train.csv."""
    rng = np.random.default_rng(0)
    raw = root / "raw"
    rows = []
    time_ids = sorted(rng.choice(np.arange(5, 32767), N_TIMES, replace=False).tolist())
    for stock in range(N_STOCKS):
        times = time_ids if stock != 3 else time_ids[:-2]  # not every stock in every bucket
        book = pd.DataFrame(
            {
                "time_id": np.repeat(times, 3),
                "seconds_in_bucket": np.tile([0, 1, 2], len(times)),
                "bid_price1": 1.0,
                "ask_price1": 1.0001,
                "bid_size1": 10,
                "ask_size1": 12,
            }
        )
        trade = book[["time_id", "seconds_in_bucket"]].assign(price=1.0, size=5, order_count=1)
        for name, frame in [("book_train.parquet", book), ("trade_train.parquet", trade)]:
            part = raw / name / f"stock_id={stock}"
            part.mkdir(parents=True)
            frame.to_parquet(part / "part.parquet", index=False)
        rows += [(stock, t, float(rng.uniform(0.001, 0.01))) for t in times]
    pd.DataFrame(rows, columns=["stock_id", "time_id", "target"]).to_csv(
        raw / "train.csv", index=False
    )
    return raw


def _run(root):
    public, private = root / "public", root / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    _prepare()(raw=_raw(root), public=public, private=private)
    return public, private


def _digest(directory):
    return {
        str(p.relative_to(directory)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_no_training_target_at_a_test_time_bucket(tmp_path):
    public, private = _run(tmp_path)
    train = pd.read_csv(public / "train.csv")
    test = pd.read_csv(public / "test.csv")
    answers = pd.read_csv(private / "test.csv")
    test_times = set(test.time_id)
    # the old split had every test time bucket in train.csv, with the other stocks' targets
    assert not test_times & set(train.time_id)
    assert not set(test.stock_id) & set(train.stock_id)
    # a quarter of the time buckets, all of the held-out stocks' rows there
    assert len(test_times) == N_TIMES // 4
    all_times = set(train.time_id) | test_times
    assert len(all_times) == N_TIMES
    assert set(test.stock_id) == {int(p.name.split("=")[1]) for p in public.glob("book_test*/*")}
    raw = pd.read_csv(tmp_path / "raw" / "train.csv")
    expected = raw[raw.stock_id.isin(test.stock_id) & raw.time_id.isin(test_times)]
    assert len(test) == len(expected) == len(answers)
    assert list(test.row_id) == list(answers.row_id)
    assert (test.row_id == test.stock_id.astype(str) + "-" + test.time_id.astype(str)).all()
    # the training stocks' rows at the other time buckets, unchanged
    kept = raw[~raw.stock_id.isin(test.stock_id) & ~raw.time_id.isin(test_times)]
    pd.testing.assert_frame_equal(train.reset_index(drop=True), kept.reset_index(drop=True))
    sample = pd.read_csv(public / "sample_submission.csv")
    assert list(sample.row_id) == list(answers.row_id) and (sample.target == 0).all()


def test_book_and_trade_data_keep_every_time_bucket(tmp_path):
    public, _ = _run(tmp_path)
    test = pd.read_csv(public / "test.csv")
    raw = tmp_path / "raw"
    for kind in ["book", "trade"]:
        for split in ["train", "test"]:
            for part in (public / f"{kind}_{split}.parquet").glob("stock_id=*"):
                source = raw / f"{kind}_train.parquet" / part.name / "part.parquet"
                assert (part / "part.parquet").read_bytes() == source.read_bytes()
    # the training stocks' book data covers the test time buckets (features, no targets)
    book = pd.read_parquet(public / "book_train.parquet")
    assert set(test.time_id) <= set(book.time_id)
    assert (
        len(list((public / "book_train.parquet").glob("*")))
        + len(list((public / "book_test.parquet").glob("*")))
        == N_STOCKS
    )


def test_prepare_is_deterministic(tmp_path):
    first = _run(tmp_path / "a")
    second = _run(tmp_path / "b")
    for a, b in zip(first, second, strict=True):
        assert _digest(a) == _digest(b)
