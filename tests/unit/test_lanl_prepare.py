"""LANL-Earthquake-Prediction prepare: the test is Kaggle-style segments of held-out cycles.

The prepare runs on a small synthetic run with the real structure (17 cycles, 16 earthquakes),
with the segment and chunk sizes scaled down.
"""

import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LANL = importlib.import_module("timeseriesgym.competitions.LANL-Earthquake-Prediction.prepare")

SEGMENT_ROWS = 7
CHUNK_ROWS = 13
# 17 cycles, as in Kaggle's run. The 4th earthquake falls between two chunks: its last row,
# row 155, is the last row of a chunk.
CYCLE_ROWS = [12, 50, 44, 50, 61, 33, 47, 52, 39, 45, 58, 41, 36, 64, 22, 51, 19]


def _run(rows: list[int], ttf_shift: float = 0.0) -> pd.DataFrame:
    """A run with the given cycle lengths; acoustic_data is the row number, so unique."""
    ttf = np.concatenate([np.arange(n, 0, -1) * 0.0123456789 for n in rows])
    ttf[-rows[-1] :] += 9.75  # the run ends before the last cycle's earthquake
    ttf[sum(rows[:12]) :] += ttf_shift
    return pd.DataFrame(
        {"acoustic_data": np.arange(len(ttf), dtype=np.int16) - 500, "time_to_failure": ttf}
    )


def _prepare(root: Path, run: pd.DataFrame, monkeypatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(LANL, "SEGMENT_ROWS", SEGMENT_ROWS)
    monkeypatch.setattr(LANL, "CHUNK_ROWS", CHUNK_ROWS)
    raw, public, private = root / "raw", root / "public", root / "private"
    for d in (raw, public, private):
        d.mkdir(parents=True)
    run.to_csv(raw / "train.csv", index=False)
    LANL.prepare(raw, public, private)
    return raw, public, private


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.csv"))}


def test_lanl_test_is_whole_segments_of_held_out_cycles(tmp_path, monkeypatch):
    run = _run(CYCLE_ROWS)
    raw, public, private = _prepare(tmp_path, run, monkeypatch)
    ends = np.cumsum(CYCLE_ROWS)  # exclusive end row of each cycle
    assert ends[3] % CHUNK_ROWS == 0  # the jump after cycle 4 is between two chunks

    # train.csv: cycles 1-12 up to one row before the 12th earthquake, bytes as before
    old_train = run.iloc[: ends[11] - 1].to_csv(index=False)
    assert (public / "train.csv").read_text() == old_train

    answers = pd.read_csv(private / "test.csv", float_precision="round_trip")
    sample = pd.read_csv(public / "sample_submission.csv")
    assert list(answers.columns) == list(sample.columns) == ["seg_id", "time_to_failure"]
    assert sample.seg_id.tolist() == answers.seg_id.tolist() == sorted(answers.seg_id)
    assert (sample.time_to_failure == 0).all()
    assert answers.seg_id.is_unique
    assert all(re.fullmatch(r"seg_[A-Za-z0-9]{6}", s) for s in answers.seg_id)
    assert sorted(p.stem for p in (public / "test").iterdir()) == answers.seg_id.tolist()

    starts = []
    for seg_id, label in zip(answers.seg_id, answers.time_to_failure, strict=True):
        seg = pd.read_csv(public / "test" / f"{seg_id}.csv")
        assert list(seg.columns) == ["acoustic_data"]  # no labels in the public files
        assert len(seg) == SEGMENT_ROWS  # was 15,000-row windows
        start = int(seg.acoustic_data.iloc[0]) + 500
        assert (
            seg.acoustic_data.tolist()
            == run.acoustic_data.iloc[start : start + SEGMENT_ROWS].tolist()
        )
        assert label == run.time_to_failure.iloc[start + SEGMENT_ROWS - 1]  # its last row
        starts.append(start)

    # Held-out cycles only, no overlaps, no segment across an earthquake, every whole segment
    # of each cycle, the last one ending at the cycle's last row.
    starts = np.array(starts)
    assert starts.min() >= ends[11]  # the first row of cycle 13
    cycle = np.searchsorted(ends, starts, side="right")
    assert (cycle == np.searchsorted(ends, starts + SEGMENT_ROWS - 1, side="right")).all()
    for c in range(12, 17):
        mine = np.sort(starts[cycle == c])
        expected = np.arange(ends[c] - SEGMENT_ROWS, ends[c - 1] - 1, -SEGMENT_ROWS)[::-1]
        assert mine.tolist() == expected.tolist()
    assert len(answers) == sum(n // SEGMENT_ROWS for n in CYCLE_ROWS[12:])

    # The answers (in id order) are not in the order of the run.
    assert starts.tolist() != sorted(starts.tolist())


def test_lanl_prepare_is_deterministic_and_ids_follow_only_private_labels(tmp_path, monkeypatch):
    first = _prepare(tmp_path / "a", _run(CYCLE_ROWS), monkeypatch)[1].parent
    again = _prepare(tmp_path / "b", _run(CYCLE_ROWS), monkeypatch)[1].parent
    assert _files(first) == _files(again)

    # Same public signal, other test labels: other ids. The ids cannot be recomputed from the
    # public files and the code alone.
    shifted = _prepare(tmp_path / "c", _run(CYCLE_ROWS, ttf_shift=1e-3), monkeypatch)[1].parent
    assert (shifted / "public" / "train.csv").read_bytes() == (
        first / "public" / "train.csv"
    ).read_bytes()
    ids = pd.read_csv(first / "public" / "sample_submission.csv").seg_id
    assert set(pd.read_csv(shifted / "public" / "sample_submission.csv").seg_id) != set(ids)


def test_lanl_prepare_checks_the_number_of_earthquakes(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="expected 16 earthquakes"):
        _prepare(tmp_path, _run(CYCLE_ROWS[:-1]), monkeypatch)
