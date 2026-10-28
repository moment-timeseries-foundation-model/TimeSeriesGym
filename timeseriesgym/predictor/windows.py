"""Forecast windows of the predictor challenges, built with the challenges' own data code.

- time-series-library-long-horizon-forecast (`tsl`): Time-Series-Library's `data_provider(args,
  flag)` with the arguments of the challenge's prepare (seq_len 96, label_len 0, pred_len 96,
  features M, target OT). Window i of a dataset is the loader's item i: context `seq_x` (96,
  n_vars), label `seq_y`, standardized with the scaler TSL fits on the training split. The
  grader runs flag "test" on the full CSVs (the private data), which is how the answers are
  made; the local check runs flag "val" on the public CSVs, which end where the test labels
  begin (their loader is patched to keep the full files' borders and scaler).
- gift-eval-short-horizon (`gift`): the challenge's `Dataset` class, `test_data` (GluonTS split
  at -prediction_length * windows, rolling windows `distance=prediction_length`), as in the
  challenge's download_data.py. The grader runs it on the full series (the private data); the
  local check on the public series, which end before their test windows, with `windows` set to
  the test's count (the validation windows then end where the test windows begin).

The grader and the local check (a copy of this package in the public data) use the same code.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .protocol import Window

SEQ_LEN = 96
PRED_LEN = 96


@dataclass
class DatasetInfo:
    name: str
    freq: str
    n_windows: int
    out_shape: tuple
    labels: np.ndarray | None = None  # (n_windows, *out_shape); only with with_labels=True
    extra: dict | None = None


class WindowSet:
    """The windows of one run, the per-dataset forecast arrays they fill, and the upstream
    submission format those arrays are assembled into."""

    def __init__(self, challenge: str, datasets: list[DatasetInfo], windows: list[Window]):
        self.challenge, self.datasets, self.windows = challenge, datasets, windows
        self.by_name = {d.name: d for d in datasets}

    def empty(self) -> dict[str, np.ndarray]:
        return {
            d.name: np.full((d.n_windows, *d.out_shape), np.nan, dtype=np.float32)
            for d in self.datasets
        }

    def sink(self, arrays: dict[str, np.ndarray]):
        def put(w: Window, a: np.ndarray) -> None:
            arrays[w.dataset][w.slot] = a

        return put

    def labels(self) -> dict[str, np.ndarray]:
        return {d.name: d.labels for d in self.datasets}

    def to_native(self, arrays: dict[str, np.ndarray]):
        """Per-dataset arrays (forecasts or labels) in the challenge's submission and answers
        format: TSL a dict of (N, 96, n_vars) arrays (the `submission` group, as read by
        load_arbitrary_h5); GIFT-Eval a one-row compound array with one float32
        (N, prediction_length) field per dataset."""
        if self.challenge == "tsl":
            return dict(arrays)
        dt = np.dtype([(d.name, np.float32, (d.n_windows, *d.out_shape)) for d in self.datasets])
        out = np.zeros(1, dtype=dt)
        for d in self.datasets:
            out[d.name][0] = arrays[d.name]
        return out

    def subset(self, names: list[str] | None = None, limit: int | None = None) -> WindowSet:
        """Only some datasets, and at most `limit` windows of each series (the earliest)."""
        keep = [d for d in self.datasets if names is None or d.name in names]
        out_ds, out_w = [], []
        for d in keep:
            ws = [w for w in self.windows if w.dataset == d.name]
            if limit is not None:
                ws = _earliest(ws, limit)
            slots = sorted(w.slot for w in ws)
            remap = {s: i for i, s in enumerate(slots)}
            labels = d.labels[slots] if d.labels is not None else None
            out_ds.append(DatasetInfo(d.name, d.freq, len(slots), d.out_shape, labels, d.extra))
            out_w += [replace(w, slot=remap[w.slot]) for w in ws]
        return WindowSet(self.challenge, out_ds, out_w)


def _earliest(ws: list[Window], limit: int) -> list[Window]:
    by_series: dict[int, list[Window]] = {}
    for w in ws:
        by_series.setdefault(w.series, []).append(w)
    keep = []
    for s in by_series.values():
        keep += sorted(s, key=lambda w: w.t0)[:limit]
    return keep


def build(challenge: str, root: Path, flag: str = "test", meta: dict | None = None, **kw):
    """The windows of the full data in `root` (flag "test", the grader) or of the public data
    (flag "val", the local check, with `meta` = the public predictor_datasets.json)."""
    root = Path(root)
    if challenge == "tsl":
        rows = None if meta is None else {d["name"]: d["full_rows"] for d in meta["datasets"]}
        return tsl_windows(root, root / "Time-Series-Library", flag, full_rows=rows, **kw)
    if challenge == "gift":
        if meta is None:
            names = [n.strip() for n in (root / "datasets.txt").read_text().splitlines()]
            return gift_windows(root, root / "dataset.py", [n for n in names if n], **kw)
        windows = {d["name"]: d["test_windows_per_series"] for d in meta["datasets"]}
        order = [d["name"] for d in meta["datasets"]]
        return gift_windows(root, root / "dataset.py", order, windows=windows, **kw)
    raise ValueError(f"unknown predictor challenge {challenge!r}")


# --------------------------------------------------------------------------- TSL

# (directory, file stem) of the nine long-term-forecasting datasets, as the prepare finds them.
TSL_DATASETS = (
    ("ETT-small", "ETTh1"),
    ("ETT-small", "ETTh2"),
    ("ETT-small", "ETTm1"),
    ("ETT-small", "ETTm2"),
    ("electricity", "electricity"),
    ("exchange_rate", "exchange_rate"),
    ("illness", "national_illness"),
    ("traffic", "traffic"),
    ("weather", "weather"),
)
TSL_FLAGS = {"train": 0, "val": 1, "test": 2}
# The four ETT datasets share one clock (ETTh1 and ETTh2 are ETTm1 and ETTm2 sampled at the
# hour; the two transformers are measured over the same period): one timeline, so an hourly
# context never shows a 15-minute value that is still to be forecast. The other datasets are
# separate sources (their CSV dates are not a shared calendar).
TSL_TIMELINES = {"ETTh1": "ETT", "ETTh2": "ETT", "ETTm1": "ETT", "ETTm2": "ETT"}


def tsl_args(name: str, root: Path, csv_name: str):
    """The arguments of the challenge prepare's get_true_array."""

    class Args:
        pass

    a = Args()
    a.data = "custom" if "ett" not in name.lower() else name
    a.pred_len, a.seq_len, a.label_len = PRED_LEN, SEQ_LEN, 0
    a.root_path, a.data_path = str(root), csv_name
    a.features, a.target, a.embed, a.freq = "M", "OT", "-", "-"
    a.batch_size, a.num_workers = 512, 1
    a.task_name, a.seasonal_patterns = "long_term_forecast", None
    a.augmentation_ratio = 0
    return a


def tsl_borders(name: str, n_rows: int) -> tuple[list[int], list[int]]:
    """Time-Series-Library's split borders (data_provider/data_loader.py) for seq_len 96."""
    if name.startswith("ETTh") or name.startswith("ETTm"):
        k = 4 if name.startswith("ETTm") else 1
        m = 30 * 24 * k
        return [0, 12 * m - SEQ_LEN, 16 * m - SEQ_LEN], [12 * m, 16 * m, 20 * m]
    num_train, num_test = int(n_rows * 0.7), int(n_rows * 0.2)
    num_vali = n_rows - num_train - num_test
    return (
        [0, num_train - SEQ_LEN, n_rows - num_test - SEQ_LEN],
        [num_train, num_train + num_vali, n_rows],
    )


def tsl_test_start(name: str, n_rows: int) -> int:
    """Row of the first test label: everything before it is training or validation data."""
    b1, _ = tsl_borders(name, n_rows)
    return b1[2] + SEQ_LEN


@contextlib.contextmanager
def no_bytecode():
    """Imports from the data directories write no __pycache__ into them."""
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = before


@contextlib.contextmanager
def tsl_imported(repo: Path):
    """TSL's data_provider, imported from `repo`; afterwards TSL's top-level modules
    (data_provider, utils, exp, ...) are forgotten again, so that the next call can import
    another checkout and the grading process keeps no `utils` of TSL's."""
    repo = str(Path(repo).resolve())
    before = set(sys.modules)
    sys.path.insert(0, repo)
    try:
        with no_bytecode():
            from data_provider.data_factory import data_provider

        yield data_provider
    finally:
        sys.path.remove(repo)
        for key in set(sys.modules) - before:
            if any(f.startswith(repo) for f in _module_files(sys.modules[key])):
                del sys.modules[key]


def _module_files(module) -> list[str]:
    """Where a module was loaded from (its file, or a package's directories)."""
    files = [getattr(module, "__file__", None)]
    try:
        files += list(getattr(module, "__path__", None) or [])
    except TypeError:  # e.g. torch._classes, whose __path__ is not a list of directories
        pass
    return [f for f in files if isinstance(f, str)]


def tsl_windows(
    data_root: Path,
    repo: Path,
    flag: str,
    full_rows: dict[str, int] | None = None,
    names: list[str] | None = None,
    with_labels: bool = False,
) -> WindowSet:
    """The windows of TSL's `flag` split. `full_rows`: each CSV's row count before the cut
    (public data); by default the CSV's own row count (full data)."""
    import pandas as pd

    datasets, windows = [], []
    with tsl_imported(repo) as data_provider:
        for group, name in TSL_DATASETS:
            if names is not None and name not in names:
                continue
            root = Path(data_root) / group
            dates = pd.to_datetime(pd.read_csv(root / f"{name}.csv", usecols=["date"])["date"])
            n_rows = int((full_rows or {}).get(name, len(dates)))
            ds, _ = data_provider(tsl_args(name, root, f"{name}.csv"), flag)
            b1s, b2s = tsl_borders(name, n_rows)
            border1, border2 = b1s[TSL_FLAGS[flag]], b2s[TSL_FLAGS[flag]]
            n = len(ds)
            x = np.asarray(ds.data_x)
            if n <= 0 or len(x) != border2 - border1 or border2 > len(dates):
                raise ValueError(f"{name}: TSL's {flag} split is not rows {border1}..{border2}")
            freq = pd.infer_freq(dates[:1000])
            n_vars = x.shape[1]
            labels = None
            if with_labels:  # a view: label i = data_y[i + 96 : i + 192], as TSL's __getitem__
                y = np.asarray(ds.data_y)[SEQ_LEN:]
                labels = np.lib.stride_tricks.sliding_window_view(y, PRED_LEN, axis=0)[:n]
                labels = labels.transpose(0, 2, 1)
            extra = {"border1": border1, "n_rows": n_rows}
            datasets.append(DatasetInfo(name, freq, n, (PRED_LEN, n_vars), labels, extra))
            stamps = dates.dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
            clocks = dates.astype("int64").to_numpy()  # ns
            for i in range(n):
                windows.append(
                    Window(
                        dataset=name,
                        item_id=name,
                        series=0,
                        t0=border1 + i + SEQ_LEN,
                        start=str(stamps[border1 + i]),
                        freq=freq,
                        prediction_length=PRED_LEN,
                        target=x[i : i + SEQ_LEN],
                        out_shape=(PRED_LEN, n_vars),
                        slot=i,
                        timeline=TSL_TIMELINES.get(name, name),
                        clock=int(clocks[border1 + i + SEQ_LEN]),  # the first forecast step
                    )
                )
    return WindowSet("tsl", datasets, windows)


# --------------------------------------------------------------------------- GIFT-Eval


def load_gift_dataset_module(path: Path):
    """The challenge's dataset.py, imported from its path."""
    spec = importlib.util.spec_from_file_location("tsgym_gift_dataset", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with no_bytecode():
        spec.loader.exec_module(module)
    return module


def gift_windows(
    data_root: Path,
    dataset_py: Path,
    names: list[str],
    windows: dict[str, int] | None = None,
    with_labels: bool = False,
) -> WindowSet:
    """`Dataset(name, term="short").test_data` of each dataset, in its order (series by series,
    each series' windows in time order). `windows`: override Dataset.windows (local check)."""
    import pandas as pd

    module = load_gift_dataset_module(Path(dataset_py))
    datasets, out = [], []
    for name in names:
        ds = module.Dataset(name=name, term="short", storage_path=str(data_root))
        if windows is not None:
            ds.__dict__["windows"] = int(windows[name])  # cached_property
        if ds.target_dim != 1:
            raise ValueError(f"{name}: only univariate datasets are supported")
        h, w_per = int(ds.prediction_length), int(ds.windows)
        labels = []
        series = -1
        k = -1
        for k, (inp, lab) in enumerate(ds.test_data):
            if k % w_per == 0:
                series += 1
                first_t0 = len(inp["target"])
            t0 = len(inp["target"])
            if t0 != first_t0 + (k % w_per) * h or len(lab["target"]) != h:
                raise ValueError(f"{name}: test window {k} is not where the split puts it")
            period = pd.Period(inp["start"])
            start = period.start_time.strftime("%Y-%m-%dT%H:%M:%S")
            out.append(
                Window(
                    dataset=name,
                    item_id=str(inp["item_id"]),
                    series=series,
                    t0=t0,
                    start=start,
                    freq=str(ds.freq),
                    prediction_length=h,
                    target=np.asarray(inp["target"]),
                    out_shape=(h,),
                    slot=k,
                    timeline=name,  # the series of a dataset share a calendar
                    clock=int((period + t0).start_time.value),  # ns, the first forecast step
                )
            )
            if with_labels:
                labels.append(np.asarray(lab["target"], dtype=np.float32))
        n = k + 1
        if n != len(ds.hf_dataset) * w_per:
            raise ValueError(f"{name}: {n} test windows, expected {len(ds.hf_dataset) * w_per}")
        datasets.append(
            DatasetInfo(
                name,
                str(ds.freq),
                n,
                (h,),
                np.stack(labels) if with_labels else None,
                {"windows_per_series": w_per, "series": len(ds.hf_dataset)},
            )
        )
    return WindowSet("gift", datasets, out)
