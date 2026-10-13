"""check_predictor: run a predictor on validation windows of the public data, the way the grader
runs it on the hidden test windows, and score it with the challenge's metric.

    python /home/data/check_predictor.py [/home/submission/predictor] [--datasets A,B]
        [--limit N] [--timeout SEC]

The same format check (contract.py), windows code (windows.py: TSL's flag "val" on the public
CSVs; GIFT-Eval's Dataset with the test's window count on the public series), protocol and call
schedule (protocol.py), predictor process (runner.py, on a copy of the directory) and metric
(the challenge's grade.py, copied next to this file as `metric/`). Differences from the
grader: the windows (validation, not test) and where the predictor runs (here: as you, with
this Python).

The prepare copies this package into the public data as `predictor_tools/`, with config.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

from . import contract, protocol
from . import windows as pw
from .launch import PredictorJob, SubprocessLauncher, run_job, thread_env

HERE = Path(__file__).resolve().parent
CONFIG = "config.json"
METADATA = "predictor_datasets.json"


def load_grade():
    """The challenge's grade function: grade.py and the modules it imports relatively, copied
    next to this file as the package `metric`. Outside TimeSeriesGym's environment, the one name
    grade.py needs from it (InvalidSubmissionError) is provided."""
    try:
        import timeseriesgym.utils  # noqa: F401
    except ImportError:
        utils = types.ModuleType("timeseriesgym.utils")
        utils.InvalidSubmissionError = type("InvalidSubmissionError", (Exception,), {})
        utils.get_logger = logging.getLogger
        package = types.ModuleType("timeseriesgym")
        package.utils = utils
        sys.modules.setdefault("timeseriesgym", package)
        sys.modules.setdefault("timeseriesgym.utils", utils)
    return importlib.import_module(f"{__package__}.metric.grade").grade


def native_answers(ws: pw.WindowSet):
    """The windows' labels in the format of the challenge's answers file, for its grade
    function. A metric whose module `mase` has `window_baselines` (GIFT-Eval's normalized MASE)
    also needs each window's scale and seasonal naive forecast: they are computed here from the
    windows' histories, as its prepare computes them for the test windows."""
    labels = ws.to_native(ws.labels())
    try:
        baselines = importlib.import_module(f"{__package__}.metric.mase").window_baselines
    except (ImportError, AttributeError):
        return labels
    scale_dt, scales, naive = [], {}, {}
    for d in ws.datasets:
        windows = sorted((w for w in ws.windows if w.dataset == d.name), key=lambda w: w.slot)
        inputs = [{"target": w.target} for w in windows]
        view = types.SimpleNamespace(
            freq=d.freq,
            prediction_length=d.out_shape[0],
            test_data=types.SimpleNamespace(input=inputs),
        )
        scales[d.name], naive[d.name] = baselines(view)
        scale_dt.append((d.name, "<f8", scales[d.name].shape))
    out = {"target": labels, "seasonal_error": np.zeros(1, np.dtype(scale_dt))}
    out["seasonal_naive"] = ws.to_native(naive)
    for name, a in scales.items():
        out["seasonal_error"][name][0] = a
    return out


def dataset_names(meta: dict, arg: str | None, error) -> list[str] | None:
    """The names of --datasets, checked against predictor_datasets.json (None: all)."""
    if arg is None:
        return None
    valid = [d["name"] for d in meta["datasets"]]
    names = [n.strip() for n in arg.split(",") if n.strip()]
    unknown = [n for n in names if n not in valid]
    if unknown or not names:
        what = f"unknown dataset(s): {', '.join(unknown)}" if unknown else "no dataset names"
        error(f"--datasets: {what}; the datasets are: {', '.join(valid)}")
    return names


def scores(grade, ws: pw.WindowSet, forecasts: dict) -> list[tuple[str, int, float]]:
    """(dataset, windows, score) per dataset, each dataset scored alone by the challenge's grade
    function, as the grader scores the test windows."""
    rows = []
    for d in ws.datasets:
        one = ws.subset(names=[d.name])
        answers = native_answers(one)
        s = grade(submission=one.to_native({d.name: forecasts[d.name]}), answers=answers)
        rows.append((d.name, d.n_windows, float(s[d.name] if isinstance(s, dict) else s)))
    return rows


def main(argv: list[str] | None = None) -> int:
    cfg = json.loads((HERE / CONFIG).read_text())
    ap = argparse.ArgumentParser(prog="check_predictor", description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", default="/home/submission/predictor")
    ap.add_argument("--datasets", help="comma-separated dataset names (default: all)")
    ap.add_argument("--limit", type=int, help="at most N windows per series (the earliest)")
    ap.add_argument("--timeout", type=float, default=float(cfg["timeout_sec"]), help="seconds")
    ap.add_argument("--data-dir", default=str(HERE.parent), help="the public data")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")
    data = Path(args.data_dir)
    meta = json.loads((data / METADATA).read_text())
    names = dataset_names(meta, args.datasets, ap.error)
    path = Path(args.path)
    max_bytes, max_files = int(cfg["max_bytes"]), int(cfg["max_files"])
    try:
        contract.check_predictor(path, max_bytes, max_files)
    except contract.ContractError as e:
        print(f"INVALID: {e}")
        return 1
    print(f"format: {path} is a valid predictor directory")
    t = time.monotonic()
    ws = pw.build(cfg["windows"], data, "val", meta=meta, with_labels=True)
    if names is not None or args.limit:
        ws = ws.subset(names=names, limit=args.limit)
    print(f"validation windows: {len(ws.windows)} ({time.monotonic() - t:.0f} s to build)")
    forecasts = ws.empty()
    last = [0.0]

    def progress(k, calls, done, total):
        if time.monotonic() - last[0] > 10:
            last[0] = time.monotonic()
            print(f"  call {k}/{calls}, {done}/{total} windows", flush=True)

    with tempfile.TemporaryDirectory(prefix="check-predictor-") as tmp:
        work = Path(tmp) / "predictor"
        try:
            contract.copy_predictor(path, work, max_bytes, max_files)
            cpus = int(cfg["cpus"])
            job = PredictorJob(work, timeout_sec=args.timeout, cpus=cpus, env=thread_env(cpus))
            stats = run_job(
                SubprocessLauncher(sys.executable),
                job,
                ws.windows,
                ws.sink(forecasts),
                max_windows=int(cfg["max_batch_windows"]),
                max_values=int(cfg["max_batch_values"]),
                progress=progress,
            )
        except (contract.ContractError, protocol.PredictorError) as e:
            print(f"FAILED (the grader would score this predictor as invalid): {e}")
            return 1
    try:
        grade = load_grade()
        rows = scores(grade, ws, forecasts)
        overall = grade(submission=ws.to_native(forecasts), answers=native_answers(ws))
    except Exception as e:  # e.g. a metric whose answers hold more than native_answers gives
        print(f"the forecasts are complete, but the metric could not be computed here: {e!r}")
        rows = None
    if rows is not None:
        print(f"\nmetric: {cfg['metric']}, on the validation windows\n")
        print(f"{'dataset':<32} {'windows':>8} {'score':>12}")
        for name, n, score in rows:
            print(f"{name:<32} {n:>8} {score:>12.6g}")
        if not isinstance(overall, dict):
            print(f"{'all datasets':<32} {len(ws.windows):>8} {float(overall):>12.6g}")
        print("(the test windows come later in time and can score differently)")
    test = {d["name"]: d["test_windows"] for d in meta["datasets"]}
    test_total = sum(test[d.name] for d in ws.datasets)
    rate = stats["windows"] / max(stats["elapsed_s"], 1e-9)
    estimate = f"about {test_total / rate:.0f} s" if rate > 0 else "an unknown time"
    print(
        f"\ntime: {stats['elapsed_s']:.1f} s for {stats['windows']} windows in {stats['calls']} "
        f"calls (loading predict.py: {stats['startup_s']:.1f} s); at this rate the "
        f"{test_total} test windows take {estimate}; the grader stops a predictor after "
        f"{cfg['timeout_sec']} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
