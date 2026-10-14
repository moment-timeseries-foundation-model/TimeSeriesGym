"""child-mind-institute-detect-sleep-states: the metric gives the scores of Kaggle's code.

`match_detections` compared every detection with every ground truth (about 3 ms per submitted
row), and `event_detection_ap` merged the detections once per tolerance (10 copies). They now
do neither; the scores must stay bitwise the same. KAGGLE holds the two functions as they were.
"""

import importlib
import struct
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

COMPETITION = "timeseriesgym.competitions.child-mind-institute-detect-sleep-states"
metric = importlib.import_module(f"{COMPETITION}.event_detection_average_precision")
grade = importlib.import_module(f"{COMPETITION}.grade").grade

TOLERANCES = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]  # grade.py's, for both events
COLUMNS = {
    "series_id_column_name": "series_id",
    "time_column_name": "step",
    "event_column_name": "event",
    "score_column_name": "score",
}

# Kaggle's functions before the change (5e8e195), verbatim.
KAGGLE = '''
def match_detections(
    tolerance: float, ground_truths: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    """
    Match detections to ground truth events. Arguments are taken from a common event x
    tolerance x series_id evaluation group.
    """
    detections_sorted = detections.sort_values(score_column_name, ascending=False).dropna()
    is_matched = np.full_like(detections_sorted[event_column_name], False, dtype=bool)
    gts_matched = set()
    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        best_error = tolerance
        best_gt = None

        for gt in ground_truths.itertuples(index=False):
            error = abs(getattr(det, time_column_name) - getattr(gt, time_column_name))
            if error < best_error and gt not in gts_matched:
                best_gt = gt
                best_error = error

        if best_gt is not None:
            is_matched[i] = True
            gts_matched.add(best_gt)

    detections_sorted["matched"] = is_matched

    return detections_sorted


def event_detection_ap(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: dict[str, list[float]],
) -> float:
    # Ensure solution and submission are sorted properly
    solution = solution.sort_values([series_id_column_name, time_column_name])
    submission = submission.sort_values([series_id_column_name, time_column_name])

    # Extract scoring intervals.
    if use_scoring_intervals:
        intervals = (
            solution.query("event in ['start', 'end']")
            .assign(
                interval=lambda x: x.groupby([series_id_column_name, event_column_name]).cumcount()
            )
            .pivot(
                index="interval",
                columns=[series_id_column_name, event_column_name],
                values=time_column_name,
            )
            .stack(series_id_column_name)
            .swaplevel()
            .sort_index()
            .loc[:, ["start", "end"]]
            .apply(lambda x: pd.Interval(*x, closed="both"), axis=1)
        )

    # Extract ground-truth events.
    ground_truths = solution.query("event not in ['start', 'end']").reset_index(drop=True)

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts(event_column_name).to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched=False)

    # Remove detections outside of scoring intervals
    if use_scoring_intervals:
        detections_filtered = []
        for (det_group, dets), (int_group, ints) in zip(
            detections.groupby(series_id_column_name),
            intervals.groupby(series_id_column_name),
            strict=False,
        ):
            assert det_group == int_group
            detections_filtered.append(filter_detections(dets, ints))
        detections_filtered = pd.concat(detections_filtered, ignore_index=True)
    else:
        detections_filtered = detections

    # Create table of event-class x tolerance x series_id values
    aggregation_keys = pd.DataFrame(
        [
            (ev, tol, vid)
            for ev in tolerances.keys()
            for tol in tolerances[ev]
            for vid in ground_truths[series_id_column_name].unique()
        ],
        columns=[event_column_name, "tolerance", series_id_column_name],
    )

    # Create match evaluation groups: event-class x tolerance x series_id
    detections_grouped = aggregation_keys.merge(
        detections_filtered, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])
    ground_truths_grouped = aggregation_keys.merge(
        ground_truths, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])
    # Match detections to ground truth events by evaluation group
    detections_matched = []
    for key in aggregation_keys.itertuples(index=False):
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(match_detections(dets["tolerance"].iloc[0], gts, dets))
    detections_matched = pd.concat(detections_matched)

    # Compute AP per event x tolerance group
    event_classes = ground_truths[event_column_name].unique()
    ap_table = (
        detections_matched.query("event in @event_classes")
        .groupby([event_column_name, "tolerance"])
        .apply(
            lambda group: average_precision_score(
                group["matched"].to_numpy(),
                group[score_column_name].to_numpy(),
                class_counts[group[event_column_name].iat[0]],
            )
        )
    )
    # Average over tolerances, then over event classes
    mean_ap = ap_table.groupby(event_column_name).mean().sum() / len(event_classes)

    return mean_ap
'''


def _kaggle() -> dict:
    """KAGGLE's functions, defined in the metric module's namespace (its column globals)."""
    functions: dict = {}
    exec(KAGGLE, vars(metric), functions)
    return functions


def _outcome(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # compared by type: both versions must fail alike
        return type(e)


def _both(monkeypatch, fn, *args):
    """fn(*args) with the metric as it is, then with Kaggle's functions."""
    new = _outcome(fn, *args)
    with monkeypatch.context() as m:
        for name, function in _kaggle().items():
            m.setattr(metric, name, function)
        old = _outcome(fn, *args)
    return new, old


def _assert_identical(new, old) -> None:
    if isinstance(old, type):
        assert new is old
        return
    assert type(new) is type(old)
    assert struct.pack("<d", new) == struct.pack("<d", old), (new, old)


def _set_columns(monkeypatch) -> None:
    """The column globals that score() sets, for calling match_detections directly."""
    for name, value in COLUMNS.items():
        monkeypatch.setattr(metric, name, value)


def _score(solution, submission, tolerances=None):
    tolerances = tolerances or {"onset": TOLERANCES, "wakeup": TOLERANCES}
    return metric.score(solution, submission, tolerances, **COLUMNS)


def _solution(rng, series=4, nights=5, missing=0.2) -> pd.DataFrame:
    """CMI's layout: an onset and a wakeup per night; a night without a sleep window has
    neither step nor timestamp."""
    rows = []
    for s in range(series):
        for night in range(1, nights + 1):
            onset = night * 17_280 + int(rng.integers(-1_500, 1_500))
            wakeup = onset + int(rng.integers(3_000, 9_000))
            gone = rng.random() < missing
            for event, step in (("onset", onset), ("wakeup", wakeup)):
                stamp = np.nan if gone else f"t{step}"
                rows.append((f"s{s}", night, event, np.nan if gone else float(step), stamp))
    return pd.DataFrame(rows, columns=["series_id", "night", "event", "step", "timestamp"])


def _submission(rng, solution, rows=300, decimals=2) -> pd.DataFrame:
    """Detections near the events (errors around the tolerances, both events), anywhere in
    the series, and in a series without events; scores with ties. One series has no wakeup."""
    events = solution.dropna(subset=["step"])
    near = events.iloc[rng.integers(0, len(events), rows // 2)]
    steps = near["step"].to_numpy() + rng.integers(-400, 400, len(near))
    swap = rng.random(len(near)) < 0.2
    kinds = np.where(swap, np.where(near["event"] == "onset", "wakeup", "onset"), near["event"])
    far = rows - len(near)
    sub = pd.DataFrame(
        {
            "series_id": np.r_[near["series_id"], rng.choice([*events["series_id"], "zz"], far)],
            "step": np.r_[steps, rng.integers(0, 120_000, far)].astype(int),
            "event": np.r_[kinds, rng.choice(["onset", "wakeup"], far)],
            "score": rng.random(rows).round(decimals),
        }
    )
    sub = sub[~((sub["series_id"] == "s1") & (sub["event"] == "wakeup"))]
    sub = sub.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    sub.insert(0, "row_id", np.arange(len(sub)))
    return sub


@pytest.mark.parametrize("seed", range(6))
def test_scores_equal_kaggle_on_random_submissions(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    solution = _solution(rng)
    submission = _submission(rng, solution, decimals=1 + seed % 3)
    # Events without a step counted as positives (as at the pin), and dropped by grade().
    _assert_identical(*_both(monkeypatch, _score, solution, submission))
    _assert_identical(*_both(monkeypatch, grade, submission, solution))
    # float steps, with fractions
    floats = submission.assign(step=submission["step"] + rng.choice([0.0, 0.5], len(submission)))
    _assert_identical(*_both(monkeypatch, grade, floats, solution))


@pytest.mark.parametrize("tolerance", TOLERANCES)
def test_each_tolerance_equals_kaggle(monkeypatch, tolerance):
    rng = np.random.default_rng(tolerance)
    solution = _solution(rng, missing=0.0)
    submission = _submission(rng, solution)
    tolerances = {"onset": [tolerance], "wakeup": [tolerance]}
    _assert_identical(*_both(monkeypatch, _score, solution, submission, tolerances))


def test_match_detections_equals_kaggle(monkeypatch):
    """The group function itself, on ground truths in any order and with equal errors."""
    _set_columns(monkeypatch)
    rng = np.random.default_rng(7)
    for _ in range(300):
        g = int(rng.integers(1, 6))
        gts = pd.DataFrame(
            {
                "event": "onset",
                "tolerance": 60,
                "series_id": "a",
                "night": np.arange(g),
                "step": rng.choice([100.0, 120.0, 140.0, 200.0, np.nan], g),
            }
        )
        dets = pd.DataFrame(
            {
                "event": "onset",
                "tolerance": 60,
                "series_id": "a",
                "step": rng.choice(
                    [90, 95, 110, 110, 125, 130, 170, 300], int(rng.integers(0, 12))
                ),
            }
        )
        dets["score"] = rng.choice([0.1, 0.5, 0.9], len(dets))
        tolerance = np.int64(rng.choice([12, 36, 60]))
        new, old = _both(monkeypatch, metric.match_detections, tolerance, gts, dets)
        pd.testing.assert_frame_equal(new, old)


def test_equal_errors_go_to_the_first_ground_truth(monkeypatch):
    _set_columns(monkeypatch)
    solution = pd.DataFrame(
        {"series_id": "a", "night": [1, 2], "event": "onset", "step": [100.0, 120.0]}
    )
    submission = pd.DataFrame(
        {"row_id": [0, 1, 2], "series_id": "a", "step": [110, 110, 110], "event": "onset"}
    )
    submission["score"] = [0.9, 0.8, 0.7]
    new, old = _both(monkeypatch, _score, solution, submission, {"onset": [12, 36]})
    _assert_identical(new, old)
    gts = solution.iloc[::-1].assign(tolerance=12)  # the first in order, not the smallest step
    dets = submission.assign(tolerance=12)
    new, old = _both(monkeypatch, metric.match_detections, 12, gts, dets)
    pd.testing.assert_frame_equal(new, old)
    assert new["matched"].tolist() == [True, True, False]
    # The choice matters when it leaves a later detection without a ground truth: 110 takes
    # 100 (the first), so 95 finds none within 12; had it taken 120, both would match.
    blocked = submission.iloc[:2].assign(step=[110, 95])
    new, old = _both(monkeypatch, _score, solution, blocked, {"onset": [12]})
    _assert_identical(new, old)
    assert new == 0.5


def test_dense_events_with_equal_scores_equal_kaggle(monkeypatch):
    """Events closer than the tolerances, so that detections compete for them, and scores
    with ties, so that the order of tied detections (the sort Kaggle's code uses) counts."""
    rng = np.random.default_rng(11)
    steps = np.arange(40) * 15.0
    solution = pd.DataFrame(
        {
            "series_id": "a",
            "night": np.r_[np.arange(40), np.arange(40)],
            "event": ["onset"] * 40 + ["wakeup"] * 40,
            "step": np.r_[steps, steps + 7],
        }
    )
    rows = 400
    submission = pd.DataFrame(
        {
            "row_id": np.arange(rows),
            "series_id": "a",
            "step": rng.integers(-30, 640, rows),
            "event": rng.choice(["onset", "wakeup"], rows),
            "score": rng.choice([0.1, 0.2, 0.3], rows),
        }
    )
    _assert_identical(*_both(monkeypatch, _score, solution, submission))


def test_equal_ground_truth_rows_match_together(monkeypatch):
    """Kaggle's loop remembers matched rows as tuples: a row equal to a matched one is taken
    too, and a row holding a float NaN (equal to nothing) can be taken again and again."""
    _set_columns(monkeypatch)
    twins = pd.DataFrame(
        {"series_id": "a", "night": [1, 1, 2], "event": "onset", "step": [100.0, 100.0, 500.0]}
    )
    nan_night = twins.assign(night=[1.0, np.nan, 2.0])
    submission = pd.DataFrame(
        {"row_id": [0, 1, 2], "series_id": "a", "step": [100, 101, 500], "event": "onset"}
    )
    submission["score"] = [0.9, 0.8, 0.7]
    for solution in (twins, nan_night):
        _assert_identical(*_both(monkeypatch, _score, solution, submission, {"onset": [12]}))
    for gts, matched in ((twins, [True, False, True]), (nan_night, [True, True, True])):
        new, old = _both(monkeypatch, metric.match_detections, 12, gts, submission)
        pd.testing.assert_frame_equal(new, old)
        assert new["matched"].tolist() == matched


def test_edge_cases_equal_kaggle(monkeypatch):
    rng = np.random.default_rng(3)
    solution = _solution(rng)
    submission = _submission(rng, solution)
    cases = {
        "empty": submission.iloc[:0],
        "one event": submission[submission["event"] == "onset"],
        "one row": submission.iloc[:1],
        "equal scores": submission.assign(score=1.0),
        "empty cells": submission.assign(
            step=submission["step"].where(rng.random(len(submission)) > 0.1),
            extra=np.where(rng.random(len(submission)) > 0.1, 1.0, np.nan),
        ),
        "integers past 2**52": submission.assign(step=submission["step"] + 2**53),
    }
    integer_solution = solution.dropna(subset=["step"]).astype({"step": int})
    for name, sub in cases.items():
        for sol in (solution, integer_solution):
            new, old = _both(monkeypatch, _score, sol, sub)
            _assert_identical(new, old)
            assert name != "empty" or new is ValueError  # as before: no detections, no score


def test_grading_time_is_linear_in_the_rows():
    """Kaggle's loop costs about 3 ms per row: about a minute for these 20,000."""
    rng = np.random.default_rng(0)
    solution = _solution(rng, series=10, nights=20, missing=0.0)
    submission = _submission(rng, solution, rows=20_000)
    start = time.perf_counter()
    grade(submission, solution)
    assert time.perf_counter() - start < 10


def test_memory_does_not_grow_with_the_tolerances():
    """The detections were merged once per tolerance: scoring held about 21 times the
    submission's size at its peak (traced allocations). Merged once, it holds about 2."""
    rng = np.random.default_rng(0)
    solution = _solution(rng, series=10, nights=20, missing=0.0)
    submission = _submission(rng, solution, rows=50_000)
    size = submission.memory_usage(deep=True).sum()
    tracemalloc.start()
    try:
        grade(submission, solution)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 5 * size
