"""ashrae-energy-prediction prepare: a two-month gap between the training labels and the test.

The prepare runs on a small synthetic year of hourly readings with Kaggle's columns. The test is
the readings from 2016-10-31 on, as before; the training labels end on 2016-08-31, so no label
of the two months before the test is public. The weather covers the whole year.
"""

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

ASHRAE = importlib.import_module("timeseriesgym.competitions.ashrae-energy-prediction.prepare")

# A few whole days around each boundary, and one in every month.
DAYS = ["2016-01-01", "2016-04-15", "2016-08-30", "2016-08-31", "2016-09-01", "2016-09-15"]
DAYS += ["2016-10-30", "2016-10-31", "2016-11-15", "2016-12-31"]


def _hours() -> list[str]:
    return [f"{day} {h:02d}:00:00" for day in DAYS for h in range(24)]


def _raw(root):
    rng = np.random.default_rng(0)
    raw = root / "raw"
    raw.mkdir(parents=True)
    hours = _hours()
    pd.DataFrame(
        {
            "site_id": [0, 0, 1],
            "building_id": [0, 1, 2],
            "primary_use": ["Education", "Office", "Lodging/residential"],
            "square_feet": [7432, 2720, 5376],
            "year_built": [2008.0, np.nan, 1991.0],
            "floor_count": [np.nan, 3.0, np.nan],
        }
    ).to_csv(raw / "building_metadata.csv", index=False)
    meters = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 3)]  # (building_id, meter)
    pd.DataFrame(
        {
            "building_id": [b for _ in hours for b, _ in meters],
            "meter": [m for _ in hours for _, m in meters],
            "timestamp": [t for t in hours for _ in meters],
            "meter_reading": rng.gamma(2.0, 50.0, len(hours) * len(meters)).round(4),
        }
    ).to_csv(raw / "train.csv", index=False)
    pd.DataFrame(
        {
            "site_id": [s for s in (0, 1) for _ in hours],
            "timestamp": hours * 2,
            "air_temperature": rng.normal(15, 8, 2 * len(hours)).round(1),
            "dew_temperature": rng.normal(5, 6, 2 * len(hours)).round(1),
        }
    ).to_csv(raw / "weather_train.csv", index=False)
    return raw


def _run(root):
    raw = _raw(root)
    public, private = root / "public", root / "private"
    public.mkdir()
    private.mkdir()
    ASHRAE.prepare(raw=raw, public=public, private=private)
    return raw, public, private


def _digest(directory):
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def test_training_labels_end_two_months_before_the_test(tmp_path):
    raw, public, private = _run(tmp_path)
    readings = pd.read_csv(raw / "train.csv")
    train = pd.read_csv(public / "train_label.csv")
    answers = pd.read_csv(private / "test.csv")

    # the training labels: every reading up to 2016-08-31 23:00, none of September or October
    # (they used to run to 2016-10-30 23:00, right up to the test)
    assert train.timestamp.max() == "2016-08-31 23:00:00"
    expected = readings[readings.timestamp < "2016-09-01"].reset_index(drop=True)
    pd.testing.assert_frame_equal(train, expected)
    gap = readings[(readings.timestamp >= "2016-09-01") & (readings.timestamp < "2016-10-31")]
    assert len(gap) and not set(gap.timestamp) & set(train.timestamp)

    # the test, as before: every reading from 2016-10-31 on, in the raw order
    expected = readings[readings.timestamp >= "2016-10-31"].reset_index(drop=True)
    pd.testing.assert_frame_equal(answers, expected)
    assert answers.timestamp.min() == "2016-10-31 00:00:00"
    sample = pd.read_csv(public / "sample_submission.csv")
    keys = ["building_id", "meter", "timestamp"]
    pd.testing.assert_frame_equal(sample[keys], answers[keys])
    assert (sample.meter_reading == 0).all()
    assert len(train) + len(gap) + len(answers) == len(readings)


def test_weather_and_buildings_cover_the_whole_year(tmp_path):
    raw, public, _ = _run(tmp_path)
    weather = pd.read_csv(raw / "weather_train.csv")
    train = pd.read_csv(public / "train_features.csv")
    test = pd.read_csv(public / "test_features.csv")
    # the weather of the gap stays in train_features.csv; test_features.csv is the test period
    assert train.timestamp.max() == "2016-10-30 23:00:00"
    assert test.timestamp.min() == "2016-10-31 00:00:00"
    both = pd.concat([train, test]).sort_values(["site_id", "timestamp"], ignore_index=True)
    pd.testing.assert_frame_equal(both, weather)
    assert (public / "building_metadata.csv").read_bytes() == (
        raw / "building_metadata.csv"
    ).read_bytes()
    assert sorted(p.name for p in public.iterdir()) == [
        "building_metadata.csv",
        "sample_submission.csv",
        "test_features.csv",
        "train_features.csv",
        "train_label.csv",
    ]


def test_prepare_is_deterministic(tmp_path):
    first = _run(tmp_path / "a")
    second = _run(tmp_path / "b")
    for a, b in zip(first[1:], second[1:], strict=True):
        assert _digest(a) == _digest(b)


def test_description_states_the_split():
    text = (Path(ASHRAE.__file__).parent / "description.md").read_text()
    assert "the training labels cover 2016-01-01 to 2016-08-31" in text
    assert "the test set covers 2016-10-31 to 2016-12-31" in text
    assert "The two months in between (2016-09-01 to 2016-10-30) have no labels" in text
    assert (ASHRAE.TRAIN_LABELS_END, ASHRAE.TEST_START) == ("2016-09-01", "2016-10-31")
