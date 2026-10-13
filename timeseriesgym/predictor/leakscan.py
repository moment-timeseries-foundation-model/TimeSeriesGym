"""Value-overlap scan: do the public files hold runs of test-label values?

The label values of a dataset are scanned for as runs of K consecutive values: for GIFT-Eval,
each series' test windows (in time order); for TSL, each column of the full CSV from the row of
the first test label on (the labels are TSL's standardization of those rows). Every run with at
least MIN_DISTINCT distinct finite values is looked up among all runs of K consecutive values in
the public data (each GIFT-Eval series; each column of each public CSV), comparing the stored
values exactly (GIFT-Eval's float32 targets; the CSV values as parsed by pandas). Runs with few
distinct values (zeros, a constant, a value repeated) are skipped: they recur by chance.

A run found once or twice in the public data is a `hit`; one found at 3 or more places is
`recurring` (a pattern of the data, such as a daily profile, rather than a copy of the test
period). Public data that still hold the test period hit almost every run; data cut before it
should hit none.

    python -m timeseriesgym.predictor.leakscan <competition-id> [--data-dir DIR] [--public DIR]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import windows as pw

K = 16
MIN_DISTINCT = 4
CHUNK = 1 << 20
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_P = np.uint64(0x100000001B3)


def _mix(bits: np.ndarray) -> np.ndarray:
    """splitmix64's finalizer: every value to a well-spread 64-bit word."""
    z = bits.astype(np.uint64)
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def _bits(values: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(values)
    view = {4: np.uint32, 8: np.uint64, 2: np.uint16}[v.dtype.itemsize]
    return v.view(view)


def run_keys(values: np.ndarray, k: int) -> np.ndarray:
    """A 64-bit key for every run of `k` consecutive values (by their bits)."""
    n = len(values) - k + 1
    if n <= 0:
        return np.zeros(0, np.uint64)
    mixed = _mix(_bits(values))
    with np.errstate(over="ignore"):
        acc = np.zeros(n, np.uint64)
        for j in range(k):
            acc = acc * _P + mixed[j : j + n]
    return acc


def informative(values: np.ndarray, k: int, min_distinct: int) -> np.ndarray:
    """For every run of `k` consecutive values: all finite, with at least `min_distinct`
    distinct values, and at least k / 2 changes from one value to the next."""
    n = len(values) - k + 1
    if n <= 0:
        return np.zeros(0, bool)
    runs = np.lib.stride_tricks.sliding_window_view(values, k)
    out = np.empty(n, bool)
    for lo in range(0, n, CHUNK):
        chunk = runs[lo : lo + CHUNK]
        changes = (np.diff(chunk, axis=1) != 0).sum(axis=1)
        distinct = 1 + (np.diff(np.sort(chunk, axis=1), axis=1) != 0).sum(axis=1)
        ok = np.isfinite(chunk).all(axis=1) & (distinct >= min_distinct)
        out[lo : lo + CHUNK] = ok & (2 * changes >= k)
    return out


class PublicRuns:
    """The runs of `k` consecutive values of the public sequences, for lookups."""

    def __init__(self, sequences: Iterable[np.ndarray], names: list, k: int):
        keys = [run_keys(np.asarray(s), k) for s in sequences]
        self.k, self.names = k, names
        self.offsets = np.cumsum([0, *(len(x) for x in keys)])
        every = np.concatenate(keys) if keys else np.zeros(0, np.uint64)
        unique = np.unique(every, return_index=True, return_counts=True)
        self.keys, self.first, self.counts = unique

    def find(self, keys: np.ndarray) -> np.ndarray:
        """For each key: its index in self.keys, or -1."""
        if not len(self.keys):
            return np.full(len(keys), -1)
        pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return np.where(self.keys[pos] == keys, pos, -1)

    def where(self, i: int) -> list:
        """The public sequence and position of the first run with key self.keys[i]."""
        at = int(self.first[i])
        seq = int(np.searchsorted(self.offsets, at, side="right")) - 1
        return [self.names[seq], at - int(self.offsets[seq])]


def scan(
    public: PublicRuns | Iterable[np.ndarray],
    labels: Iterable[np.ndarray],
    k: int = K,
    min_distinct: int = MIN_DISTINCT,
    label_names: list | None = None,
    examples: int = 5,
) -> dict:
    """How many informative label runs occur in the public sequences (see the module
    docstring), with up to `examples` hits: [label sequence, position] and where in the
    public data the run first occurs."""
    if not isinstance(public, PublicRuns):
        public = list(public)
        public = PublicRuns(public, list(range(len(public))), k)
    k = public.k
    runs = n_informative = hits = recurring = 0
    shown = []
    for li, seq in enumerate(labels):
        seq = np.asarray(seq)
        ok = informative(seq, k, min_distinct)
        runs += len(ok)
        at = np.flatnonzero(ok)
        n_informative += len(at)
        i = public.find(run_keys(seq, k)[at])
        present = i >= 0
        many = present & (public.counts[np.maximum(i, 0)] >= 3)
        hit = present & ~many
        hits += int(hit.sum())
        recurring += int(many.sum())
        for j in np.flatnonzero(hit)[: max(0, examples - len(shown))]:
            name = label_names[li] if label_names else li
            shown.append({"label": [name, int(at[j])], "public": public.where(int(i[j]))})
    return {
        "k": k,
        "label_runs": runs,
        "informative_runs": n_informative,
        "hits": hits,
        "recurring": recurring,
        "examples": shown,
    }


# --------------------------------------------------------------------------- the challenges


def gift_table(root: Path, name: str):
    """A GIFT-Eval dataset directory's Arrow table (read with pyarrow)."""
    import pyarrow as pa

    (arrow,) = sorted((Path(root) / name).glob("*.arrow"))
    with pa.memory_map(str(arrow)) as source:
        return pa.ipc.open_stream(source).read_all()


def gift_series(root: Path, name: str) -> list[np.ndarray]:
    """The targets of a GIFT-Eval dataset directory, as stored."""
    target = gift_table(root, name).column("target").combine_chunks()
    return [target[i].values.to_numpy(zero_copy_only=False) for i in range(len(target))]


def gift_scan(public: Path, private: Path, k: int = K, min_distinct: int = MIN_DISTINCT) -> dict:
    """Per dataset: each series' test windows (the full series from its first test window)
    against the public series of every dataset in the public directory."""
    names = _gift_names(private)
    test = pw.gift_windows(private, private / "dataset.py", names)
    first = {}
    for w in test.windows:
        first.setdefault((w.dataset, w.series), w.t0)
    public_series, public_names = [], []
    for name in names:
        public_series += gift_series(public, name)
        public_names += [f"{name} {i}" for i in gift_table(public, name).column("item_id")]
    indexes, out = {}, {}
    for d in test.datasets:
        full = gift_series(private, d.name)
        labels = [s[first[(d.name, i)] :] for i, s in enumerate(full)]
        kk = min(k, min(len(s) for s in labels))
        if kk not in indexes:
            indexes[kk] = PublicRuns(public_series, public_names, kk)
        items = [f"{d.name} {i}" for i in gift_table(private, d.name).column("item_id")]
        out[d.name] = scan(indexes[kk], labels, min_distinct=min_distinct, label_names=items)
    return out


def _gift_names(root: Path) -> list[str]:
    return [n.strip() for n in (Path(root) / "datasets.txt").read_text().splitlines() if n.strip()]


def csv_columns(path: Path) -> dict[str, np.ndarray]:
    """The numeric columns of a CSV (all but `date`), as pandas parses them."""
    import pandas as pd

    frame = pd.read_csv(path)
    return {c: frame[c].to_numpy(np.float64) for c in frame.columns if c != "date"}


def tsl_scan(public: Path, private: Path, k: int = K, min_distinct: int = MIN_DISTINCT) -> dict:
    """Per dataset: each column of the full CSV from the first test label's row on, against
    every column of every public CSV."""
    columns, names = [], []
    for group, name in pw.TSL_DATASETS:
        for col, values in csv_columns(Path(public) / group / f"{name}.csv").items():
            columns.append(values)
            names.append(f"{name} {col}")
    index = PublicRuns(columns, names, k)
    del columns
    out = {}
    for group, name in pw.TSL_DATASETS:
        full = csv_columns(Path(private) / group / f"{name}.csv")
        start = pw.tsl_test_start(name, len(next(iter(full.values()))))
        labels = [c[start:] for c in full.values()]
        label_names = [f"{name} {c} (from row {start})" for c in full]
        out[name] = scan(index, labels, min_distinct=min_distinct, label_names=label_names)
    return out


SCANS = {"gift": gift_scan, "tsl": tsl_scan}


def main(argv: list[str] | None = None) -> int:
    from timeseriesgym.predictor.grading import settings
    from timeseriesgym.registry import registry

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("competition_id")
    ap.add_argument("--data-dir", default=str(registry.get_data_dir()))
    ap.add_argument("--public", help="the public data to scan (default: the prepared public)")
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args(argv)
    competition = registry.set_data_dir(Path(args.data_dir)).get_competition(args.competition_id)
    public = Path(args.public) if args.public else competition.public_dir
    report = SCANS[settings(competition)["windows"]](public, competition.private_dir, args.k)
    print(json.dumps(report, indent=2))
    return 0 if all(r["hits"] == 0 for r in report.values()) else 1  # a hit: look at it


if __name__ == "__main__":
    raise SystemExit(main())
