"""GIFT-Eval's point-forecast score, as on its leaderboard: on each dataset, the MASE of the
forecasts divided by the MASE of the seasonal naive forecast, and the geometric mean of these
ratios over the datasets (https://huggingface.co/spaces/Salesforce/GIFT-Eval, `src/utils.py`).

GIFT-Eval scores a dataset with gluonts' `evaluate_model(..., metrics=[MASE()], axis=None,
mask_invalid_label=True, allow_nan_forecast=False, seasonality=get_seasonality(freq))`. Its
seasonal naive forecast is statsforecast's `SeasonalNaive(season_length=get_seasonality(freq))`
on each window's history, after gluonts' `LastValueImputation` has filled the missing values
(`notebooks/naive.ipynb` in https://github.com/SalesforceAIResearch/gift-eval).

`window_baselines` runs when the challenge is prepared and needs gluonts. Grading needs only
numpy and scipy.
"""

import numpy as np
from scipy.stats import gmean


def seasonal_naive(history: np.ndarray, season: int, horizon: int) -> np.ndarray:
    """statsforecast's `SeasonalNaive`: the last `season` values of each series in `history`
    (time on the last axis), repeated over the horizon. As in GIFT-Eval, missing values are
    filled first, in float32, with gluonts' `LastValueImputation`."""
    from gluonts.transform.feature import LastValueImputation

    history = np.asarray(history, dtype=np.float32)
    if history.shape[-1] < season:  # statsforecast forecasts NaN, which GIFT-Eval rejects
        raise ValueError(f"A history of {history.shape[-1]} values is shorter than {season}.")
    forecasts = []
    for series in history.reshape(-1, history.shape[-1]):
        if np.isnan(series).any():
            series = LastValueImputation()(series.copy())
        forecasts.append(np.resize(series[-season:], horizon))
    return np.stack(forecasts).reshape(history.shape[:-1] + (horizon,))


def window_baselines(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Each test window's MASE scale and seasonal naive forecast, from the window's history, for
    a `dataset.Dataset`. The scale is gluonts' `seasonal_error`: the mean absolute difference
    between the history and itself shifted by the season (1 if the history is shorter), over
    the pairs of observed values; NaN if there is no such pair. Returns arrays shaped like the
    label windows without, and with, the time axis."""
    from gluonts.ev.ts_stats import seasonal_error
    from gluonts.time_feature import get_seasonality

    season = get_seasonality(dataset.freq)
    scales, forecasts = [], []
    for entry in dataset.test_data.input:
        history = np.ma.masked_invalid(np.asarray(entry["target"], dtype=np.float64))
        scale = seasonal_error(history, seasonality=season, time_axis=-1)
        scales.append(np.ma.filled(scale, np.nan)[..., 0])
        forecasts.append(seasonal_naive(entry["target"], season, dataset.prediction_length))
    return np.stack(scales), np.stack(forecasts)


def mase(target: np.ndarray, forecast: np.ndarray, scale: np.ndarray) -> float:
    """gluonts' `MASE` over all windows of a dataset (`axis=None`): the mean of
    |target - forecast| / scale over the observed target values, with one scale per window
    (`scale` has the shape of `target` without the time axis). gluonts divides masked arrays,
    so windows whose scale is 0 or NaN do not count; here the same entries are left out."""
    error = np.abs(np.asarray(target, np.float64) - np.asarray(forecast, np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = error / np.asarray(scale, np.float64)[..., None]
    counted = np.isfinite(scaled)  # a missing target gives NaN, a zero scale inf or NaN
    return float(scaled[counted].mean()) if counted.any() else float("nan")


def normalized_mase(ratios: list[float]) -> float:
    """The geometric mean of the per-dataset ratios MASE / seasonal naive MASE."""
    with np.errstate(divide="ignore"):  # a ratio of 0 (a dataset forecast exactly) gives 0
        return float(gmean(np.asarray(ratios, dtype=np.float64)))
