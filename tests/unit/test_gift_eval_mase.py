"""gift-eval-short-horizon scores GIFT-Eval's leaderboard metric: on each dataset, MASE divided by
the seasonal naive forecast's MASE, and the geometric mean of these ratios over the datasets."""

import hashlib
import importlib
import io
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_percentage_error

from timeseriesgym.utils import InvalidSubmissionError, load_answers

COMPETITION = "timeseriesgym.competitions.gift-eval-short-horizon"


def _module(name: str):
    return importlib.import_module(f"{COMPETITION}.{name}")


def _grade(submission, answers):
    return _module("grade").grade(submission, answers)


def _compound(arrays: dict, dtype=np.float32) -> np.ndarray:
    data = np.zeros(1, dtype=[(name, dtype, array.shape) for name, array in arrays.items()])
    for name, array in arrays.items():
        data[name][0] = array
    return data


def _answers(targets: dict, scales: dict, naive: dict) -> dict:
    """The prepared `labels` group, as `load_answers` returns it."""
    return {
        "target": _compound(targets),
        "seasonal_error": _compound(scales, np.float64),
        "seasonal_naive": _compound(naive),
    }


def _toy_answers():
    rng = np.random.default_rng(0)
    targets = {
        "a/H": rng.uniform(10, 20, (3, 4)).astype(np.float32),
        "b/W": rng.uniform(0, 5, (2, 6)).astype(np.float32),
    }
    scales = {"a/H": np.array([1.0, 2.0, 0.5]), "b/W": np.array([3.0, 1.5])}
    naive = {name: target + rng.uniform(-2, 2, target.shape) for name, target in targets.items()}
    return targets, scales, naive


def test_a_perfect_forecast_scores_0_and_the_seasonal_naive_forecast_1():
    targets, scales, naive = _toy_answers()
    answers = _answers(targets, scales, naive)

    assert _grade(_compound(targets), answers) == 0.0
    assert _grade(_compound(naive), answers) == 1.0
    worse = {name: forecast + 3 for name, forecast in naive.items()}
    assert _grade(_compound(worse), answers) > 1.0


def test_the_score_is_the_geometric_mean_of_the_mase_ratios():
    targets = {"a/H": np.full((2, 3), 10, np.float32), "b/W": np.full((1, 4), 5, np.float32)}
    scales = {"a/H": np.array([2.0, 2.0]), "b/W": np.array([1.0])}
    naive = {name: target + 1 for name, target in targets.items()}  # MASE 0.5 and 1
    answers = _answers(targets, scales, naive)
    forecast = {"a/H": targets["a/H"] + 4, "b/W": targets["b/W"] - 1}  # MASE 2 and 1

    # ratios 4 and 1: geometric mean 2 (their arithmetic mean would be 2.5)
    assert _grade(_compound(forecast), answers) == pytest.approx(2.0, rel=1e-12)


def test_actual_values_of_zero_no_longer_explode():
    # hierarchical_sales has weeks without sales; MAPE divided by max(|0|, eps)
    targets = {"sales/W": np.array([[0, 3, 0, 1], [2, 0, 0, 4]], np.float32)}
    scales = {"sales/W": np.array([1.5, 2.0])}
    naive = {"sales/W": np.array([[2, 2, 2, 2], [1, 1, 1, 1]], np.float32)}
    answers = _answers(targets, scales, naive)
    forecast = np.full((2, 4), 1.0, np.float32)

    errors = np.abs(targets["sales/W"] - forecast) / scales["sales/W"][:, None]
    naive_errors = np.abs(targets["sales/W"] - naive["sales/W"]) / scales["sales/W"][:, None]
    expected = errors.mean() / naive_errors.mean()
    assert _grade(_compound({"sales/W": forecast}), answers) == pytest.approx(expected)
    assert expected < 2
    mape = mean_absolute_percentage_error(targets["sales/W"].ravel(), forecast.ravel())
    assert mape > 1e15  # the previous score


def test_missing_actual_values_and_windows_without_a_scale_do_not_count():
    target = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, 9]], np.float32)
    answers = _answers(
        {"a/H": target},
        {"a/H": np.array([1.0, 0.0, np.nan])},  # gluonts masks x / 0 and x / NaN
        {"a/H": np.zeros((3, 3), np.float32)},
    )
    forecast = np.array([[2, 0, 3], [0, 0, 0], [0, 0, 0]], np.float32)

    # counted: (0, 0) and (0, 2): MASE (1 + 0) / 2, seasonal naive (1 + 3) / 2
    assert _grade(_compound({"a/H": forecast}), answers) == pytest.approx(0.25)
    forecast[0, 1] = forecast[1:] = 1e6  # entries that do not count
    assert _grade(_compound({"a/H": forecast}), answers) == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_forecasts_are_invalid(bad):
    targets, scales, naive = _toy_answers()
    forecast = {name: target.copy() for name, target in targets.items()}
    forecast["b/W"][1, 2] = bad  # gluonts would drop an infinite entry from the mean
    with pytest.raises(InvalidSubmissionError):
        _grade(_compound(forecast), _answers(targets, scales, naive))


def test_missing_fields_and_wrong_shapes_are_invalid():
    targets, scales, naive = _toy_answers()
    answers = _answers(targets, scales, naive)
    with pytest.raises(InvalidSubmissionError, match="no field"):
        _grade(_compound({"a/H": targets["a/H"]}), answers)
    with pytest.raises(InvalidSubmissionError, match=r"prediction shape \(2, 5\)"):
        _grade(_compound({"a/H": targets["a/H"], "b/W": targets["b/W"][:, :5]}), answers)
    with pytest.raises(ValueError, match="prepare"):  # answers prepared before this metric
        _grade(_compound(targets), _compound(targets))


# ---------------------------------------------------------------- agreement with gluonts


def _gluonts_test_data(series: list, freq: str, horizon: int, windows: int):
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.split import split

    start = pd.Period("2020-01-01 00:00", freq=freq)
    dataset = ListDataset([{"start": start, "target": s} for s in series], freq=freq)
    _, template = split(dataset, offset=-horizon * windows)
    return template.generate_instances(prediction_length=horizon, windows=windows, distance=horizon)


def _gluonts_mase(test_data, forecast: np.ndarray, season: int) -> float:
    from gluonts.dataset.util import forecast_start
    from gluonts.ev.metrics import MASE
    from gluonts.model.evaluation import evaluate_forecasts
    from gluonts.model.forecast import QuantileForecast

    forecasts = [
        QuantileForecast(
            forecast_arrays=forecast[i][None],
            forecast_keys=["0.5"],
            start_date=forecast_start(entry),
        )
        for i, entry in enumerate(test_data.input)
    ]
    result = evaluate_forecasts(  # as GIFT-Eval calls evaluate_model
        forecasts,
        test_data=test_data,
        metrics=[MASE()],
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season,
    )
    return float(result["MASE[0.5]"].iloc[0])


def test_mase_and_its_scale_agree_with_gluonts():
    pytest.importorskip("gluonts")
    from gluonts.time_feature import get_seasonality

    mase = _module("mase")
    rng = np.random.default_rng(1)
    hours = np.arange(200)
    series = [10 + 5 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1, 200) for _ in range(3)]
    series[1][[5, 6, 7, 100, 150, 181, 190]] = np.nan  # gaps in the history and in the labels
    series.append(np.full(200, 4.0))  # a constant series: scale 0, left out by gluonts too
    test_data = _gluonts_test_data(series, "h", horizon=12, windows=2)
    dataset = SimpleNamespace(freq="h", prediction_length=12, test_data=test_data)

    scale, naive = mase.window_baselines(dataset)
    target = np.stack([entry["target"] for entry in test_data.label])
    assert scale.shape == (8,) and naive.shape == target.shape == (8, 12)
    assert np.isnan(target).any() and (scale == 0).sum() == 2
    forecast = (target + rng.normal(0, 2, target.shape)).astype(np.float32)
    forecast[np.isnan(forecast)] = 0.0

    season = get_seasonality("h")
    assert season == 24
    for f in (forecast, naive):
        expected = _gluonts_mase(test_data, f, season)
        assert mase.mase(target, f, scale) == pytest.approx(expected, rel=1e-5)


def test_seasonal_naive_repeats_the_last_season_after_filling_gaps():
    pytest.importorskip("gluonts")
    from gluonts.time_feature import get_seasonality

    seasonal_naive = _module("mase").seasonal_naive
    history = np.array([1, 2, 3, 4, 5, np.nan, 7], np.float32)
    # LastValueImputation: the gap takes the last value before it (5)
    np.testing.assert_array_equal(seasonal_naive(history, 3, 5), [5, 5, 7, 5, 5])
    leading = np.array([np.nan, np.nan, 2, 3], np.float32)
    np.testing.assert_array_equal(seasonal_naive(leading, 4, 6), [2, 2, 2, 3, 2, 2])
    with pytest.raises(ValueError, match="shorter"):  # statsforecast would forecast NaN
        seasonal_naive(history, 8, 2)

    # The seasons of the challenge's datasets (frequencies as in their data)
    seasons = {f: get_seasonality(f) for f in ("5min", "h", "D", "W-WED", "W-FRI")}
    assert seasons == {"5min": 288, "h": 24, "D": 1, "W-WED": 1, "W-FRI": 1}


# ---------------------------------------------------------------- prepare

# One small series set per dataset in datasets.txt: (frequency, length)
TOY_DATASETS = {
    "LOOP_SEATTLE/5T": ("5T", 700),
    "hierarchical_sales/W": ("W-WED", 60),
    "kdd_cup_2018_with_missing/H": ("H", 500),
    "m4_daily": ("D", 50),
    "solar/W": ("W-FRI", 40),
}


def _toy_raw_dir(raw: Path) -> None:
    datasets = pytest.importorskip("datasets")
    features = datasets.Features(
        {
            "item_id": datasets.Value("string"),
            "start": datasets.Value("timestamp[s]"),
            "freq": datasets.Value("string"),
            "target": datasets.Sequence(datasets.Value("float32")),
        }
    )
    rng = np.random.default_rng(2)
    for name, (freq, length) in TOY_DATASETS.items():
        targets = rng.uniform(0, 10, (2, length)).astype(np.float32)
        if "missing" in name:
            targets[0, rng.choice(length, 40, replace=False)] = np.nan
        rows = {
            "item_id": ["0", "1"],
            "start": [datetime(2020, 1, 1)] * 2,
            "freq": [freq] * 2,
            "target": [t.tolist() for t in targets],
        }
        datasets.Dataset.from_dict(rows, features=features).save_to_disk(str(raw / name))

    # raw/test.h5 as download_data.py writes it
    dataset_class = _module("dataset").Dataset
    labels = {}
    for name in TOY_DATASETS:
        dataset = dataset_class(name=name, term="short", to_univariate=False, storage_path=raw)
        labels[name] = np.array([entry["target"] for entry in dataset.test_data.label])
    with h5py.File(raw / "test.h5", "w") as f:
        f.create_dataset("label", data=_compound(labels))


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_prepare_keeps_each_windows_scale_and_seasonal_naive_forecast(tmp_path):
    pytest.importorskip("gluonts")
    raw = tmp_path / "raw"
    _toy_raw_dir(raw)
    shutil.copytree(raw, tmp_path / "raw2")
    runs = []
    for run in ("one", "two"):
        public, private = tmp_path / run / "public", tmp_path / run / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        _module("prepare").prepare(raw if run == "one" else tmp_path / "raw2", public, private)
        runs.append((public, private))
    (public, private), (public2, private2) = runs

    # deterministic: the same bytes twice
    assert _md5(private / "test.h5") == _md5(private2 / "test.h5")
    assert _md5(public / "sample_submission.h5") == _md5(public2 / "sample_submission.h5")

    answers = load_answers(private / "test.h5")
    assert set(answers) == {"target", "seasonal_error", "seasonal_naive"}
    with h5py.File(raw / "test.h5") as f:  # prepare moves the datasets, not test.h5
        raw_labels = f["label"][()]
    names = list(TOY_DATASETS)
    assert list(answers["target"].dtype.names) == names
    for name in names:
        np.testing.assert_array_equal(answers["target"][name][0], raw_labels[name][0])

    # solar/W: weekly, season 1: the last value of each window's history
    dataset_class = _module("dataset").Dataset
    solar = dataset_class(name="solar/W", term="short", to_univariate=False, storage_path=public)
    histories = [entry["target"] for entry in solar.test_data.input]
    np.testing.assert_array_equal(
        answers["seasonal_naive"]["solar/W"][0],
        np.stack([np.full(solar.prediction_length, h[-1]) for h in histories]),
    )
    np.testing.assert_allclose(
        answers["seasonal_error"]["solar/W"][0],
        [np.abs(np.diff(h.astype(np.float64))).mean() for h in histories],
    )
    # LOOP_SEATTLE/5T: season 288, a day of 5-minute steps
    loop = dataset_class(
        name="LOOP_SEATTLE/5T", term="short", to_univariate=False, storage_path=public
    )
    history = next(iter(loop.test_data.input))["target"]
    np.testing.assert_array_equal(
        answers["seasonal_naive"]["LOOP_SEATTLE/5T"][0][0], history[-288:][: loop.prediction_length]
    )

    # graded through the loaders: seasonal naive 1.0; the zero sample is valid
    assert _grade(answers["seasonal_naive"], answers) == 1.0
    with h5py.File(public / "sample_submission.h5") as f:
        sample = f["submission"][()]
    assert np.isfinite(_grade(sample, answers))


# ---------------------------------------------------------------- leaderboard


def _published(model: str, mase: dict, leakage: str | None = "No") -> tuple:
    """One model directory of the GIFT-Eval space: its all_results.csv rows and config.json."""
    configurations = _module("leaderboard").CONFIGURATIONS
    rows = pd.DataFrame(
        {
            "dataset": [configurations[name] for name in mase] + ["m4_weekly/W/short"],
            "model": model,
            "eval_metrics/MASE[0.5]": [*mase.values(), 9.0],
        }
    )
    config = {"model": model}
    if leakage is not None:
        config["testdata_leakage"] = leakage
    return rows, config


def test_the_leaderboard_scores_published_results_with_this_metric():
    leaderboard = _module("leaderboard")
    names = list(leaderboard.CONFIGURATIONS)
    reference = dict(zip(names, [0.8, 1.0, 1.3, 3.3, 1.5], strict=True))
    ratios = {"good": [0.5] * 5, "mixed": [4, 1, 1, 1, 0.5]}
    results = {
        "Seasonal_Naive": _published("Seasonal_Naive", reference),
        **{
            model: _published(model, {n: r * reference[n] for n, r in zip(names, rs, strict=True)})
            for model, rs in ratios.items()
        },
        "leaky": _published("leaky", {n: 0.1 for n in names}, leakage="Yes"),
        "unflagged": _published("unflagged", {n: 0.1 for n in names}, leakage=None),
        "partial": _published("partial", {n: 0.1 for n in names[:4]}),
        "nan": _published("nan", {**{n: 0.1 for n in names[:4]}, names[4]: float("nan")}),
    }

    built = leaderboard.build(results)
    assert list(built.columns) == ["teamId", "teamName", "submissionDate", "score"]
    assert built["teamName"].tolist() == ["good", "Seasonal_Naive", "mixed"]
    assert built["score"].tolist() == [0.5, 1.0, 1.1487]  # mixed: 2 ** (1 / 5)
    assert built["teamId"].tolist() == [0, 1, 2]
    assert (built["submissionDate"] == leaderboard.REVISION_DATE).all()


def test_the_leaderboard_is_read_at_the_pinned_revision(tmp_path, monkeypatch):
    leaderboard = _module("leaderboard")
    files = ["results/A b (+c)/all_results.csv", "results/A b (+c)/config.json", "README.md"]
    files += ["results/grouped_results_by_domain.csv", "results/A b (+c)/notes.txt"]
    opened = []

    def urlopen(url):
        opened.append(url)
        body = (
            json.dumps({"siblings": [{"rfilename": f} for f in files]}) if "/api/" in url else url
        )
        return io.BytesIO(body.encode())

    monkeypatch.setattr(leaderboard.urllib.request, "urlopen", urlopen)
    leaderboard.fetch_results(tmp_path)
    revision = leaderboard.REVISION
    assert opened == [
        f"https://huggingface.co/api/spaces/Salesforce/GIFT-Eval/revision/{revision}",
        f"https://huggingface.co/spaces/Salesforce/GIFT-Eval/resolve/{revision}/"
        "results/A%20b%20%28%2Bc%29/all_results.csv",
        f"https://huggingface.co/spaces/Salesforce/GIFT-Eval/resolve/{revision}/"
        "results/A%20b%20%28%2Bc%29/config.json",
    ]
    assert sorted(p.name for p in (tmp_path / "A b (+c)").iterdir()) == [
        "all_results.csv",
        "config.json",
    ]


def test_the_shipped_leaderboard_ranks_gift_evals_models():
    from timeseriesgym.data import get_leaderboard
    from timeseriesgym.registry import registry

    competition = registry.get_competition("gift-eval-short-horizon")
    leaderboard = get_leaderboard(competition)
    assert len(leaderboard) == 123  # of 143 models; the other 20 are flagged for leakage
    scores = dict(zip(leaderboard["teamName"], leaderboard["score"], strict=True))
    # Naive's published MASE gives the score the grader gives the naive forecast (1.02262)
    assert scores["Seasonal_Naive"] == 1.0 and scores["Naive"] == 1.02262
    assert not {"TimesFM", "Chronos_large", "MoiraiAgent-leaking"} & set(scores)

    grader = competition.grader
    assert grader.is_lower_better(leaderboard)
    info = grader.rank_score(1.0, leaderboard)  # 123 teams: gold is rank 10
    assert info["gold_threshold"] == 0.71687 and info["silver_threshold"] == 0.74775
    assert info["bronze_threshold"] == 0.76343 and info["median_threshold"] == 0.77337
    assert not (info["gold_medal"] or info["silver_medal"] or info["bronze_medal"])
    assert grader.rank_score(0.7, leaderboard)["gold_medal"]
