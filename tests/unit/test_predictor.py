"""The predictor protocol of gift-eval-short-horizon and time-series-library-long-horizon-forecast
(timeseriesgym/predictor): the round schedule, the wire format, the predictor process, the
directory format, and both challenges end to end on small synthetic data, with their own
prepare.py (TSL with a minimal stand-in for Time-Series-Library's data loader).
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from timeseriesgym.data import generate_checksums
from timeseriesgym.grade import grade_jsonl, grade_predictor, validate_submission
from timeseriesgym.predictor import contract, leakscan, protocol
from timeseriesgym.predictor import grading as pg
from timeseriesgym.predictor import windows as pw
from timeseriesgym.predictor.launch import PredictorJob, SubprocessLauncher, run_job
from timeseriesgym.registry import Registry
from timeseriesgym.utils import load_answers

REPO = Path(__file__).resolve().parents[2]
COMPETITIONS = REPO / "timeseriesgym" / "competitions"
GIFT = "gift-eval-short-horizon"
TSL = "time-series-library-long-horizon-forecast"


def _window(dataset="d", series=0, t0=10, ctx=4, h=2, shape=None, slot=0):
    target = np.arange(t0 - ctx, t0, dtype=np.float32)
    return protocol.Window(
        dataset, f"s{series}", series, t0, "2020-01-01T00:00:00", "h", h, target,
        shape or (h,), slot,
    )  # fmt: skip


# ---------------------------------------------------------------- schedule (leakage)


def _revealed_before_commit(rounds: list[list[protocol.Window]]):
    """The first window any value of whose forecast period was shown before it was forecast.
    A context ends at its window's t0; it covers [t0 - len(target), t0)."""
    seen: dict = {}
    for rnd in rounds:
        for w in rnd:  # the whole call is revealed before any forecast of the call is kept
            lo, hi = w.t0 - len(w.target), w.t0
            seen.setdefault((w.dataset, w.series), []).append((lo, hi))
        for w in rnd:
            for lo, hi in seen[(w.dataset, w.series)]:
                if lo < w.t0 + w.prediction_length and hi > w.t0:
                    return w
    return None


def test_schedule_tsl_like_stride_one_never_shows_a_label_early():
    # TSL: stride-1 windows with a 96-step context; any two of them in one call would leak.
    ws = [_window("a", 0, 96 + i, ctx=96, h=96, slot=i) for i in range(300)]
    ws += [_window("b", 0, 96 + i, ctx=96, h=96, slot=i) for i in range(120)]
    rng = np.random.default_rng(0)
    rng.shuffle(ws)
    rounds = protocol.schedule(ws)
    assert len(rounds) == 300 and max(len(r) for r in rounds) == 2
    assert _revealed_before_commit(rounds) is None
    # Counter-check: one big call (the order of TSL's test loader) leaks.
    assert _revealed_before_commit([ws]) is not None


def test_schedule_gift_like_full_history_never_shows_a_label_early():
    # GIFT-Eval: W non-overlapping windows at the end of each series; context = all before.
    ws = []
    for s, (n, w, h) in enumerate([(100, 5, 8), (60, 5, 8), (40, 1, 8)]):
        for k in range(w):
            t0 = n - (w - k) * h
            ws.append(_window("g", s, t0, ctx=t0, h=h, slot=len(ws)))
    rounds = protocol.schedule(ws)
    assert [len(r) for r in rounds] == [3, 2, 2, 2, 2]
    assert _revealed_before_commit(rounds) is None
    assert _revealed_before_commit([ws]) is not None  # gluonts' all-at-once order leaks


def _clocked(dataset, t0, minutes_per_step, timeline):
    """A window on a shared clock: steps of `minutes_per_step` minutes, 96 in, 96 out."""
    w = _window(dataset, 0, t0, ctx=96, h=96, slot=t0)
    w.timeline, w.clock, w.meta = timeline, t0 * minutes_per_step, {"step": minutes_per_step}
    return w


def _clock_leak(rounds):
    """A value shown (in minutes) inside the forecast period of a window not yet committed,
    across the series of one timeline."""
    shown: dict = {}
    for rnd in rounds:
        for w in rnd:
            st = w.meta["step"]
            shown.setdefault(w.timeline, []).append((w.clock - 96 * st, w.clock))
        for w in rnd:
            end = w.clock + 96 * w.meta["step"]
            if any(lo < end and hi > w.clock for lo, hi in shown[w.timeline]):
                return w
    return None


def test_schedule_shared_clock_never_shows_a_related_label_early():
    # ETTh1 is ETTm1 sampled at the hour: hourly contexts reach 4x further in time per window.
    m = [_clocked("ETTm1", 1000 + i, 15, "ETT") for i in range(400)]
    h = [_clocked("ETTh1", 250 + i, 60, "ETT") for i in range(100)]
    rounds = protocol.schedule(m + h)
    assert len(rounds) == 400 and _clock_leak(rounds) is None
    assert all(len(r) in (1, 2) for r in rounds)  # an hourly window every fourth round
    # Per-dataset rounds (no shared timeline) leak through the hourly series.
    for w in m + h:
        w.timeline = w.dataset
    rounds = [[*a, *b] for a, b in zip(protocol.schedule(m), protocol.schedule(h), strict=False)]
    for w in m + h:
        w.timeline = "ETT"  # judge the leak on the shared clock
    assert _clock_leak(rounds) is not None


def test_tsl_ett_datasets_share_one_clock():
    assert {pw.TSL_TIMELINES[n] for n in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")} == {"ETT"}
    assert all(name not in pw.TSL_TIMELINES for _, name in pw.TSL_DATASETS[4:])


def test_schedule_rejects_two_windows_with_the_same_start():
    with pytest.raises(ValueError, match="two windows"):
        protocol.schedule([_window(t0=10), _window(t0=10, slot=1)])


def test_batches_split_rounds_by_windows_and_values():
    ws = [_window("d", s, 10, ctx=4, slot=s) for s in range(10)]
    assert [len(b) for b in protocol.batches(ws, max_windows=4, max_values=10**6)] == [4, 4, 2]
    assert [len(b) for b in protocol.batches(ws, max_windows=100, max_values=9)] == [2] * 5
    big = [_window("d", 0, 100, ctx=50)]
    assert [len(b) for b in protocol.batches(big, 10, 8)] == [1]  # one window may exceed


# ---------------------------------------------------------------- wire format and runner


def test_encode_decode_roundtrip():
    arrays = [np.arange(6, dtype="<f4").reshape(2, 3), np.zeros(0, dtype="<f8"), np.ones(2)]
    frame = b"".join(bytes(b) for b in protocol.encode({"type": "x"}, arrays))
    payload = bytearray(frame[8:])
    assert protocol.header_of(payload) == {"type": "x"}
    specs = [
        {"dtype": "<f4", "shape": [2, 3]},
        {"dtype": "<f8", "shape": [0]},
        {"dtype": "<f8", "shape": [2]},
    ]
    got = protocol.decode(payload, specs)
    assert np.array_equal(got[0], arrays[0]) and got[1].size == 0
    assert np.array_equal(got[2], [1.0, 1.0])
    with pytest.raises(ValueError, match="longer"):
        protocol.decode(payload, specs[:2])  # trailing bytes
    with pytest.raises(ValueError, match="shorter"):
        protocol.decode(payload, [*specs, {"dtype": "<f4", "shape": [1]}])


def _reply(arrays, header=None):
    header = header or {
        "type": "forecasts",
        "arrays": [{"dtype": a.dtype.str, "shape": list(a.shape)} for a in arrays],
    }
    return bytearray(b"".join(bytes(b) for b in protocol.encode(header, arrays))[8:])


def test_replies_are_checked():
    batch = [_window(h=3, shape=(3,)), _window(t0=20, h=3, shape=(3,), slot=1)]
    good = [np.ones(3, np.float32), np.ones(3, np.float64)]
    assert len(protocol._check_reply(_reply(good), batch)) == 2
    bad = {
        "count": [np.ones(3, np.float32)],
        "shape": [np.ones(3, np.float32), np.ones(4, np.float32)],
        "nan": [np.ones(3, np.float32), np.array([1, np.nan, 2], np.float32)],
        "dtype": [np.ones(3, np.float32), np.ones(3, np.int64)],
        "float32 range": [np.ones(3, np.float32), np.full(3, 1e300)],
    }
    for arrays in bad.values():
        with pytest.raises(protocol.PredictorError):
            protocol._check_reply(_reply(arrays), batch)
    with pytest.raises(protocol.PredictorError, match="predict\\(\\) failed: boom"):
        protocol._check_reply(_reply([], {"type": "error", "error": "boom"}), batch)
    with pytest.raises(protocol.PredictorError, match="unreadable"):
        protocol._check_reply(bytearray(b"\xff\xff\xff\x7f{}"), batch)


def _predictor(root: Path, source: str) -> Path:
    d = root / "predictor"
    d.mkdir(parents=True)
    (d / "predict.py").write_text(source)
    return d


def _run_local(predictor: Path, windows, timeout=60.0, max_windows=3):
    got = {}
    stats = run_job(
        SubprocessLauncher(),
        PredictorJob(predictor, timeout_sec=timeout),
        windows,
        lambda w, a: got.__setitem__(w.slot, a),
        max_windows=max_windows,
    )
    return got, stats


LAST_VALUE = """
import numpy as np
print("prints go to the log, not into the protocol")
def predict(windows):
    print("called with", len(windows))
    return [np.repeat(w["target"][-1:], w["prediction_length"], axis=0) for w in windows]
"""


def test_runner_serves_batches_in_rounds(tmp_path):
    ws = [
        _window("d", s, 10 + k * 2, ctx=10 + k * 2, slot=s * 3 + k)
        for s in range(4)
        for k in range(3)
    ]
    got, stats = _run_local(_predictor(tmp_path, LAST_VALUE), ws)
    assert stats["windows"] == 12 and stats["calls"] == 6  # 3 rounds of 4, at most 3 per call
    for w in ws:
        assert np.array_equal(got[w.slot], np.full(2, w.target[-1]))


@pytest.mark.parametrize(
    "source, match",
    [
        ("raise RuntimeError('at import')\n", "loading predict.py failed"),
        ("x = 1\n", "no function predict"),
        ("def predict(windows):\n    raise ValueError('in predict')\n", "in predict"),
        ("def predict(windows):\n    return []\n", "0 forecasts"),
        ("import numpy as np\ndef predict(w):\n    return [np.zeros(5) for _ in w]\n", "shape"),
        ("import os\ndef predict(windows):\n    os._exit(3)\n", "exited"),
        (
            "import os\ndef predict(windows):\n    os.write(4, b'garbage!' * 4)\n    return []\n",
            "bytes|unreadable|exited|forecasts",
        ),
    ],
)
def test_runner_failures_are_predictor_errors(tmp_path, source, match):
    with pytest.raises(protocol.PredictorError, match=match):
        _run_local(_predictor(tmp_path, source), [_window()])


def test_runner_timeout(tmp_path):
    src = "import time\ndef predict(windows):\n    time.sleep(30)\n"
    t = time.monotonic()
    with pytest.raises(protocol.PredictorError, match="timed out"):
        _run_local(_predictor(tmp_path, src), [_window()], timeout=2)
    assert time.monotonic() - t < 10


def test_runner_accepts_a_stacked_array_and_ints(tmp_path):
    src = "import numpy as np\ndef predict(windows):\n    return np.ones((len(windows), 2), int)\n"
    got, _ = _run_local(_predictor(tmp_path, src), [_window(), _window(series=1, slot=1)])
    assert got[0].dtype == np.float64 and np.array_equal(got[1], [1.0, 1.0])


def test_launcher_stops_the_predictor_and_its_children(tmp_path):
    pid_file = tmp_path / "child.pid"
    src = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "def predict(windows):\n    return [[0.0] * w['prediction_length'] for w in windows]\n"
    )
    _run_local(_predictor(tmp_path, src), [_window()])
    pid = int(pid_file.read_text())
    for _ in range(50):  # the child is killed with the process group; wait until it is reaped
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        with open(f"/proc/{pid}/stat") as f:
            assert f.read().split()[2] == "Z", "the predictor's child is still running"


def test_launcher_gives_the_predictor_a_minimal_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TSGYM_SECRET_FOR_TEST", "x")
    src = (
        "import os\nimport numpy as np\n"
        "def predict(windows):\n"
        "    v = float('TSGYM_SECRET_FOR_TEST' in os.environ) + 10 * float(os.getcwd() == "
        "os.path.dirname(os.path.abspath(__file__)))\n"
        "    return [np.full(w['prediction_length'], v) for w in windows]\n"
    )
    got, _ = _run_local(_predictor(tmp_path, src), [_window()])
    assert np.array_equal(got[0], [10.0, 10.0])  # no inherited variables; cwd = the predictor


# ---------------------------------------------------------------- directory format


def test_predictor_contract_valid_and_copy(tmp_path):
    pred = _predictor(tmp_path, "def predict(windows):\n    return []\n")
    (pred / "pkg").mkdir()
    (pred / "pkg" / "w.bin").write_bytes(b"\x00" * 10)
    contract.check_predictor(pred)
    contract.copy_predictor(pred, tmp_path / "copy")
    assert (tmp_path / "copy" / "pkg" / "w.bin").read_bytes() == b"\x00" * 10
    for src in ("from model import predict\n", "predict = lambda w: w\n", "import x as predict\n"):
        (pred / "predict.py").write_text(src)
        contract.check_predictor(pred)


@pytest.mark.parametrize(
    "case, match",
    [
        ("missing", "missing file"),
        ("no_function", "must define"),
        ("syntax", "not valid Python"),
        ("symlink", "only regular files"),
        ("too_many", "more than 3 files"),
        ("too_big", "more than 100 bytes"),
        ("not_a_dir", "must be a directory"),
    ],
)
def test_predictor_contract_rejects(tmp_path, case, match):
    pred = _predictor(tmp_path, "def predict(windows):\n    return []\n")
    if case == "missing":
        (pred / "predict.py").unlink()
    elif case == "no_function":
        (pred / "predict.py").write_text("def forecast(windows):\n    return []\n")
    elif case == "syntax":
        (pred / "predict.py").write_text("def predict(:\n")
    elif case == "symlink":
        (pred / "link").symlink_to(tmp_path)
    elif case == "too_many":
        for i in range(3):
            (pred / f"f{i}").write_text("")
    elif case == "too_big":
        (pred / "w.bin").write_bytes(b"\x00" * 200)
    elif case == "not_a_dir":
        pred = pred / "predict.py"
    with pytest.raises(contract.ContractError, match=match):
        contract.check_predictor(pred, max_bytes=100, max_entries=3)


# ---------------------------------------------------------------- value-overlap scan


def test_leakscan_finds_copied_label_runs_and_skips_uninformative_ones():
    rng = np.random.default_rng(0)
    public = [rng.normal(size=500).astype(np.float32) for _ in range(3)]
    labels = [rng.normal(size=40).astype(np.float32) for _ in range(4)]
    clean = leakscan.scan(public, labels, k=8)
    assert clean["informative_runs"] == 4 * 33 and clean["hits"] == clean["recurring"] == 0
    leaky = leakscan.scan([*public, np.concatenate(labels)], labels, k=8)  # the old public data
    assert leaky["hits"] == leaky["informative_runs"] == 4 * 33
    # zeros, constants and NaN runs are skipped: they recur by chance
    dull = [np.zeros(20, np.float32), np.full(20, 3.0, np.float32), np.full(20, np.nan)]
    assert leakscan.scan(dull, dull, k=8)["informative_runs"] == 0
    # so are low-resolution readings that stay put (4 distinct values, 4 changes in 16 steps)
    reading = np.array([0.4, 0.4, *[0.3] * 8, 0.4, 0.4, 0.4, 0.4, 0.5, 0.6], np.float32)
    assert leakscan.scan([reading], [reading], k=16)["informative_runs"] == 0
    # a pattern found at 3 or more places of the public data is recurring, not a hit
    motif = rng.normal(size=8).astype(np.float32)
    recurring = leakscan.scan([np.tile(motif, 3)], [motif], k=8)
    assert recurring == {**recurring, "hits": 0, "recurring": 1}


# ---------------------------------------------------------------- GIFT-Eval, end to end


def _competition(data: Path, cid: str, competition_dir: Path):
    """The competition as registered, with its data in `data`; its prepare_fn is the copy of
    the challenge's prepare.py in `competition_dir` (which reads that directory's files),
    imported as a module of a package made of that directory, so that its relative imports
    resolve there."""
    import dataclasses
    import types

    name = f"toy_{competition_dir.parent.name}_{cid.replace('-', '_')}"
    package = types.ModuleType(name)
    package.__path__ = [str(competition_dir)]
    sys.modules[name] = package
    module = importlib.import_module(f"{name}.prepare")
    competition = Registry(data).get_competition(cid)
    return dataclasses.replace(competition, prepare_fn=module.prepare), module


def _prepare(competition) -> None:
    for d in (competition.raw_dir, competition.public_dir, competition.private_dir):
        d.mkdir(parents=True, exist_ok=True)
    competition.prepare_fn(competition.raw_dir, competition.public_dir, competition.private_dir)


def _hf_dataset(path: Path, freq: str, lengths: list[int], rng, nan_at=None) -> None:
    import datasets as hf

    targets = []
    for n in lengths:
        t = (10 + np.cumsum(rng.normal(0, 1, n))).astype(np.float32)
        if nan_at is not None:
            t[nan_at] = np.nan
        targets.append(t)
    rows = {
        "item_id": [f"i{k}" for k in range(len(lengths))],
        "start": [pd.Timestamp("2020-01-01")] * len(lengths),
        "freq": [freq] * len(lengths),
        "target": targets,
    }
    features = hf.Features(
        {
            "item_id": hf.Value("string"),
            "start": hf.Value("timestamp[s]"),
            "freq": hf.Value("string"),
            "target": hf.Sequence(hf.Value("float32")),
        }
    )
    hf.Dataset.from_dict(rows, features=features).save_to_disk(str(path))


# m4_daily: 1 window per series (3); toy/H: ceil(0.1 * 600 / 48) = 2 windows per series (4).
GIFT_TOY = {"m4_daily": ("D", [70, 64, 90]), "toy/H": ("H", [600, 600])}
GIFT_WINDOWS = 7


@pytest.fixture(scope="module")
def gift(tmp_path_factory):
    """gift-eval-short-horizon prepared on toy data with its own prepare.py; the raw data are
    laid out as download_data.py lays them out (test.h5 = Dataset(...).test_data.label), with a
    sibling frequency (toy/D) that is not in datasets.txt."""
    root = tmp_path_factory.mktemp("gift")
    cdir = root / "competition"
    shutil.copytree(COMPETITIONS / GIFT, cdir)
    (cdir / "datasets.txt").write_text("".join(f"{n}\n" for n in GIFT_TOY))
    competition, _ = _competition(root / "data", GIFT, cdir)
    raw = competition.raw_dir
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for name, (freq, lengths) in {**GIFT_TOY, "toy/D": ("D", [200, 200])}.items():
        _hf_dataset(raw / name, freq, lengths, rng, nan_at=590 if freq == "H" else None)
    dataset_module = pw.load_gift_dataset_module(cdir / "dataset.py")
    dt, labels = [], {}
    for name in [*GIFT_TOY, "toy/D"]:
        ds = dataset_module.Dataset(name=name, term="short", storage_path=str(raw))
        labels[name] = np.array([x["target"] for x in ds.test_data.label])
        dt.append((name, np.float32, labels[name].shape))
    data = np.zeros(1, dtype=np.dtype(dt))
    for name in labels:
        data[name][0] = labels[name]
    with h5py.File(raw / "test.h5", "w") as f:
        f.create_dataset("label", data=data)
    _prepare(competition)
    return competition, labels


def _perfect_predictor(root: Path, ws: pw.WindowSet, answers) -> Path:
    """A predictor that looks each test window's answers up (missing answers, which the metric
    ignores, as 0): what a predictor holding the answers would score."""
    pred = root / "perfect"
    pred.mkdir()
    index, table = {}, {}
    labels = pg.label_arrays(ws, answers)
    for w in ws.windows:
        rows = labels[w.dataset]
        key = f"{w.dataset}|{w.item_id}|{w.start}|{len(w.target)}"
        index[key] = f"k{len(index)}"
        table[index[key]] = np.nan_to_num(rows[w.slot])
    np.savez(pred / "answers.npz", **table)
    (pred / "index.json").write_text(json.dumps(index))
    (pred / "predict.py").write_text(
        "import json\nfrom pathlib import Path\nimport numpy as np\n"
        "HERE = Path(__file__).parent\n"
        "TABLE = np.load(HERE / 'answers.npz')\n"
        "INDEX = json.loads((HERE / 'index.json').read_text())\n"
        "def predict(windows):\n"
        "    keys = [f\"{w['dataset']}|{w['item_id']}|{w['start']}|{len(w['target'])}\" "
        "for w in windows]\n"
        "    return [TABLE[INDEX[k]] for k in keys]\n"
    )
    return pred


def test_gift_test_windows_are_the_old_answers_in_order(gift):
    competition, old = gift
    ws = pg.test_windows(competition, with_labels=True)
    assert [d.name for d in ws.datasets] == list(GIFT_TOY)
    assert len(ws.windows) == GIFT_WINDOWS
    for d in ws.datasets:  # Dataset(...).test_data.label of the full data, as download_data.py
        assert np.array_equal(d.labels, old[d.name], equal_nan=True)
    pg.check_windows(ws, load_answers(competition.answers))


def test_gift_answers_may_hold_more_than_the_labels(gift):
    """Answers that also hold a metric's per-window scales (a group with `target`, as for
    GIFT-Eval's normalized MASE) are read the same way."""
    competition, _ = gift
    ws = pg.test_windows(competition, with_labels=True)
    labels = pg.label_arrays(ws, load_answers(competition.answers))
    compound = ws.to_native(labels)
    group = {"target": compound, "seasonal_error": np.zeros(1), "seasonal_naive": compound}
    for answers in (compound, group):
        pg.check_windows(ws, answers)
        for name, a in pg.label_arrays(ws, answers).items():
            assert np.array_equal(a, labels[name], equal_nan=True)


def test_gift_public_data_hold_no_test_label(gift):
    competition, _ = gift
    public, private = competition.public_dir, competition.private_dir
    assert not (public / "toy" / "D").exists() and not (private / "toy" / "D").exists()
    assert not (public / "sample_submission.h5").exists()
    ws = pg.test_windows(competition)
    first = {}
    for w in ws.windows:
        first.setdefault((w.dataset, w.series), w.t0)
    for name in GIFT_TOY:  # each public series is its full series up to its first test window
        full = leakscan.gift_series(private, name)
        cut = leakscan.gift_series(public, name)
        for s, (a, b) in enumerate(zip(full, cut, strict=True)):
            assert len(b) == first[(name, s)] and np.array_equal(a[: len(b)], b, equal_nan=True)
    report = leakscan.gift_scan(public, private, k=8)
    assert all(r["informative_runs"] > 0 and r["hits"] == 0 for r in report.values()), report
    # positive control: the full series (what the public data used to be) hit every run
    leaky = leakscan.gift_scan(private, private, k=8)
    assert all(r["hits"] == r["informative_runs"] > 0 for r in leaky.values()), leaky
    meta = json.loads((public / "predictor_datasets.json").read_text())
    assert [d["test_windows"] for d in meta["datasets"]] == [3, 4]
    assert meta["test_rounds"] == len(protocol.schedule(ws.windows))


def test_gift_perfect_predictor_scores_perfectly_and_sample_is_valid(gift, tmp_path):
    competition, _ = gift
    ws = pg.test_windows(competition)
    perfect = _perfect_predictor(tmp_path, ws, load_answers(competition.answers))
    report = grade_predictor(perfect, competition)
    assert report.valid_submission and report.score == 0.0
    sample = grade_predictor(competition.public_dir / "sample_predictor", competition)
    assert sample.valid_submission and sample.score > 0.5
    # the same score as the forecasts submitted as a static submission (unchanged metric)
    zeros = {d.name: np.zeros((d.n_windows, *d.out_shape), np.float32) for d in ws.datasets}
    assert sample.score == pg.score_forecasts(competition, ws, zeros)


def test_gift_grades_with_any_launcher(gift):
    """The launcher is the only part that starts the predictor (a sandbox implements it): it
    gets the job, with the private paths the predictor must not read."""
    competition, _ = gift
    jobs = []

    class Recording(SubprocessLauncher):
        def __call__(self, job):
            jobs.append(job)
            return super().__call__(job)

    sample = competition.public_dir / "sample_predictor"
    assert grade_predictor(sample, competition, Recording()).valid_submission
    (job,) = jobs
    assert competition.private_dir in job.protected and job.env["OMP_NUM_THREADS"] == "8"
    # it ran the grader's own copy of the directory, which is gone afterwards
    assert job.predictor_dir != sample and not job.predictor_dir.exists()


@pytest.mark.parametrize(
    "source",
    [
        None,  # no predictor directory
        "def predict(windows):\n    raise RuntimeError('boom')\n",
        "import numpy as np\ndef predict(w):\n    return [np.full(x['prediction_length'], np.nan) "
        "for x in w]\n",
        "def forecast(windows):\n    return []\n",
    ],
)
def test_gift_failing_predictors_are_invalid(gift, tmp_path, source):
    competition, _ = gift
    path = tmp_path / "predictor" if source is None else _predictor(tmp_path, source)
    report = grade_predictor(path, competition)
    assert not report.valid_submission and report.score is None


def test_gift_grade_jsonl_dispatches_predictors(gift, tmp_path):
    competition, _ = gift
    lines = [
        {
            "competition_id": GIFT,
            "submission_path": str(competition.public_dir / "sample_predictor"),
        }
    ]
    (tmp_path / "s.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    registry = Registry(competition.public_dir.parents[2])
    grade_jsonl(tmp_path / "s.jsonl", tmp_path / "out", registry)
    (report,) = json.loads(next((tmp_path / "out").glob("*.json")).read_text())[
        "competition_reports"
    ]
    assert report["valid_submission"] and float(report["score"]) > 0.5


def test_gift_validate_submission_checks_the_format_only(gift, tmp_path):
    competition, _ = gift
    sample = competition.public_dir / "sample_predictor"
    assert validate_submission(sample, competition)[0]
    assert validate_submission(sample / "predict.py", competition)[0]  # an upload of predict.py
    marker = tmp_path / "ran"
    side_effect = f"open({str(marker)!r}, 'w')\ndef predict(windows):\n    return []\n"
    assert validate_submission(_predictor(tmp_path, side_effect) / "predict.py", competition)[0]
    assert not marker.exists()  # parsed, never run
    (tmp_path / "submission.h5").write_bytes(b"\x89HDF")
    valid, message = validate_submission(tmp_path / "submission.h5", competition)
    assert not valid and "predictor" in message


def test_gift_local_check_runs_on_validation_windows(gift, tmp_path):
    competition, _ = gift
    public = competition.public_dir
    proc = subprocess.run(
        [sys.executable, str(public / "check_predictor.py"), str(public / "sample_predictor")],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"validation windows: {GIFT_WINDOWS}" in proc.stdout and "toy/H" in proc.stdout
    assert "all datasets" in proc.stdout and competition.grader.name in proc.stdout
    assert "floor" not in proc.stdout and "reward" not in proc.stdout
    assert not list(public.rglob("__pycache__"))  # the public data stay as prepared
    bad = _predictor(tmp_path, "def predict(windows):\n    raise ValueError('nope')\n")
    proc = subprocess.run(
        [sys.executable, str(public / "check_predictor.py"), str(bad)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 1 and "nope" in proc.stdout


FAKE_MASE = """
import numpy as np

def window_baselines(dataset):
    # the check's view of the validation windows: their histories, in the answers' order
    scales = [float(len(e["target"])) for e in dataset.test_data.input]
    naive = [np.full(dataset.prediction_length, e["target"][-1]) for e in dataset.test_data.input]
    return np.array(scales), np.stack(naive)
"""
FAKE_GRADE = """
import numpy as np
from .mase import window_baselines  # noqa: F401

def grade(submission, answers):
    total = 0.0
    for name in answers["target"].dtype.names:
        t, s = answers["target"][name][0], answers["seasonal_error"][name][0]
        total += float(np.nanmean(np.abs(submission[name][0] - t) / s[:, None]))
        naive = answers["seasonal_naive"][name][0]
        assert np.array_equal(naive[:, 0], naive[:, -1], equal_nan=True)
    return total
"""


def test_gift_local_check_builds_the_answers_of_a_scaled_metric(gift, tmp_path):
    """A metric whose answers also hold each window's scale and seasonal naive forecast (as
    GIFT-Eval's normalized MASE, with its `mase.window_baselines`) gets them from the
    validation windows' histories."""
    competition, _ = gift
    public = tmp_path / "public"
    shutil.copytree(competition.public_dir, public)
    (public / "predictor_tools" / "metric" / "grade.py").write_text(FAKE_GRADE)
    (public / "predictor_tools" / "metric" / "mase.py").write_text(FAKE_MASE)
    proc = subprocess.run(
        [sys.executable, str(public / "check_predictor.py"), str(public / "sample_predictor")],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=300,
    )
    assert proc.returncode == 0 and "could not be computed" not in proc.stdout, proc.stdout
    meta = json.loads((public / "predictor_datasets.json").read_text())
    ws = pw.build("gift", public, "val", meta=meta, with_labels=True)
    expected = 0.0
    for d in ws.datasets:  # zero forecasts, each window scaled by its history's length
        lengths = {w.slot: len(w.target) for w in ws.windows if w.dataset == d.name}
        scale = np.array([lengths[i] for i in range(d.n_windows)], float)
        expected += float(np.nanmean(np.abs(d.labels) / scale[:, None]))
    assert f"{'all datasets':<32} {GIFT_WINDOWS:>8} {expected:>12.6g}" in proc.stdout


def test_gift_prepare_is_deterministic(gift, tmp_path):
    competition, _ = gift
    cdir = tmp_path / "competition"
    shutil.copytree(COMPETITIONS / GIFT, cdir)
    (cdir / "datasets.txt").write_text("".join(f"{n}\n" for n in GIFT_TOY))
    again, _ = _competition(tmp_path / "data", GIFT, cdir)
    shutil.copytree(competition.raw_dir, again.raw_dir)
    _prepare(again)
    for a, b in (
        (competition.public_dir, again.public_dir),
        (competition.private_dir, again.private_dir),
    ):
        assert generate_checksums(a) == generate_checksums(b)


# ---------------------------------------------------------------- TSL long horizon, end to end

# A minimal stand-in for Time-Series-Library's data loader: the same borders, scaler and items
# (and the same text where the public copy is patched).
FAKE_TSL = {
    "data_provider/__init__.py": "",
    "data_provider/data_factory.py": """\
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
}


def data_provider(args, flag):
    data_set = data_dict[args.data](
        args, args.root_path, flag, [args.seq_len, args.label_len, args.pred_len],
        args.features, args.data_path, args.target,
    )
    print(flag, len(data_set))
    return data_set, DataLoader(data_set, batch_size=args.batch_size, shuffle=flag != "test")
""",
    "data_provider/data_loader.py": """\
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


class _Base(Dataset):
    def __init__(self, args, root_path, flag, size, features, data_path, target):
        self.args, self.root_path, self.data_path = args, root_path, data_path
        self.target = target
        self.seq_len, self.label_len, self.pred_len = size
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.__read_data__()

    def _split(self, df_data, border1s, border2s, border1, border2):
        self.scaler = StandardScaler().fit(df_data[border1s[0]:border2s[0]].values)
        data = self.scaler.transform(df_data.values)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        mark = np.zeros(1)
        return self.data_x[index:s_end], self.data_y[r_begin:r_end], mark, mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_ETT_hour(_Base):
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self._split(df_raw[df_raw.columns[1:]], border1s, border2s, border1, border2)


class Dataset_ETT_minute(_Base):
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        m = 30 * 24 * 4
        border1s = [0, 12 * m - self.seq_len, 16 * m - self.seq_len]
        border2s = [12 * m, 16 * m, 20 * m]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self._split(df_raw[df_raw.columns[1:]], border1s, border2s, border1, border2)


class Dataset_Custom(_Base):
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self._split(df_raw[df_raw.columns[1:]], border1s, border2s, border1, border2)
""",
    "dataset/ignored.csv": "a\n1\n",  # removed by the prepare, as upstream's .csv and .json
}
# ETTh1 needs TSL's fixed borders (14,400 rows); weather is a `custom` dataset.
TSL_TOY = (("ETT-small", "ETTh1", 14_500, "h"), ("weather", "weather", 1_500, "10min"))


def _toy_csv(path: Path, rows: int, freq: str, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"date": pd.date_range("2016-07-01", periods=rows, freq=freq)})
    for col in ("HUFL", "MUFL", "OT"):
        frame[col] = np.round(10 + np.cumsum(rng.normal(0, 1, rows)), 3)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame.to_csv(path, index=False)


@pytest.fixture(scope="module")
def tsl(tmp_path_factory):
    """time-series-library-long-horizon-forecast prepared on toy data with its own prepare.py;
    its two downloads (a Google Drive folder of zips, the TSL repository) give local files."""
    root = tmp_path_factory.mktemp("tsl")
    source = root / "source"
    rng = np.random.default_rng(2)
    for group, name, rows, freq in TSL_TOY:
        _toy_csv(source / group / f"{name}.csv", rows, freq, rng)
    shutil.copytree(COMPETITIONS / TSL, root / "competition")
    competition, module = _competition(root / "data", TSL, root / "competition")

    def download_folder(url, output, quiet=False, use_cookies=False):
        for group, name, _, _ in TSL_TOY:
            with zipfile.ZipFile(Path(output) / f"{group}.zip", "w") as z:
                z.write(source / group / f"{name}.csv", f"{group}/{name}.csv")
        return [str(p) for p in Path(output).glob("*.zip")]

    def clone_repository(url, destination, branch=None, commit=None):
        for rel, text in FAKE_TSL.items():
            (destination / rel).parent.mkdir(parents=True, exist_ok=True)
            (destination / rel).write_text(text)
        return destination

    patch = pytest.MonkeyPatch()
    patch.setattr(module.gdown, "download_folder", download_folder)
    patch.setattr(module, "clone_repository", clone_repository)
    patch.setattr(pw, "TSL_DATASETS", tuple((g, n) for g, n, _, _ in TSL_TOY))
    _prepare(competition)
    yield competition, module, source
    patch.undo()


def test_tsl_test_windows_are_the_old_answers_in_order(tsl):
    competition, module, source = tsl
    ws = pg.test_windows(competition, with_labels=True)
    assert [d.n_windows for d in ws.datasets] == [2785, 205]
    with pw.tsl_imported(competition.private_dir / "Time-Series-Library"):
        for group, name, _, _ in TSL_TOY:  # the old prepare: batch_y of TSL's test loader
            old = module.get_true_array(name, root_path=source / group, data_path=f"{name}.csv")
            assert np.array_equal(ws.by_name[name].labels, old), name
    pg.check_windows(ws, load_answers(competition.answers))


def test_tsl_public_data_hold_no_test_label(tsl):
    competition, _, source = tsl
    public, private = competition.public_dir, competition.private_dir
    assert not (public / "sample_submission.h5").exists()
    assert not list(private.rglob("ignored.csv")) and not list(public.rglob("ignored.csv"))
    for group, name, rows, _ in TSL_TOY:
        full = (source / group / f"{name}.csv").read_bytes().splitlines(keepends=True)
        cut = (public / group / f"{name}.csv").read_bytes().splitlines(keepends=True)
        start = pw.tsl_test_start(name, rows)
        assert len(cut) == 1 + start and cut == full[: 1 + start]  # up to the first test label
    report = leakscan.tsl_scan(public, private)
    assert all(r["informative_runs"] > 0 and r["hits"] == 0 for r in report.values()), report
    leaky = leakscan.tsl_scan(private, private)  # positive control: the full CSVs
    assert all(r["hits"] == r["informative_runs"] > 0 for r in leaky.values()), leaky
    loader = (public / "Time-Series-Library" / "data_provider" / "data_loader.py").read_text()
    assert "TSG_FULL_ROWS" in loader and '"weather.csv": 1500' in loader


def test_tsl_public_loader_keeps_the_full_files_splits(tsl):
    competition, _, _ = tsl
    meta = json.loads((competition.public_dir / "predictor_datasets.json").read_text())
    full = pw.build("tsl", competition.private_dir, "val", with_labels=True)
    public = pw.build("tsl", competition.public_dir, "val", meta=meta, with_labels=True)
    for a, b in zip(full.datasets, public.datasets, strict=True):
        assert a.n_windows == b.n_windows and np.array_equal(a.labels, b.labels)
    pairs = zip(full.windows, public.windows, strict=True)
    assert all(np.array_equal(a.target, b.target) for a, b in pairs)
    # flag "test" on the public data gives the validation split
    with pw.tsl_imported(competition.public_dir / "Time-Series-Library") as data_provider:
        root = competition.public_dir / "weather"
        ds, _ = data_provider(pw.tsl_args("weather", root, "weather.csv"), "test")
    assert len(ds) == public.by_name["weather"].n_windows


def test_tsl_perfect_predictor_scores_perfectly(tsl, tmp_path):
    competition, _, _ = tsl
    ws = pg.test_windows(competition)
    perfect = _perfect_predictor(tmp_path, ws, load_answers(competition.answers))
    report = grade_predictor(perfect, competition)
    assert report.valid_submission and report.score == {"ETTh1": 0.0, "weather": 0.0}
    sample = grade_predictor(competition.public_dir / "sample_predictor", competition)
    assert sample.valid_submission and all(v > 0.1 for v in sample.score.values())


def test_tsl_local_check_runs_on_validation_windows(tsl):
    competition, _, _ = tsl
    public = competition.public_dir
    script = (  # the public copy of the tools, with the toy's datasets
        f"import sys\nsys.path.insert(0, {str(public)!r})\n"
        "import predictor_tools.windows as tw\n"
        f"tw.TSL_DATASETS = {tuple((g, n) for g, n, _, _ in TSL_TOY)!r}\n"
        "from predictor_tools.check import main\n"
        f"sys.exit(main([{str(public / 'sample_predictor')!r}, '--limit', '50']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "validation windows: 100" in proc.stdout and "weather" in proc.stdout


# ---------------------------------------------------------------- registry, descriptions, tools


@pytest.mark.parametrize("cid", [GIFT, TSL])
def test_predictor_challenges_take_a_predictor(cid, tmp_path):
    competition = Registry(tmp_path).get_competition(cid)
    assert competition.predictor_config["windows"] == {GIFT: "gift", TSL: "tsl"}[cid]
    assert competition.sample_submission.name == "predict.py"
    text = competition.description
    for needle in ("/home/submission/predictor/", "predict(windows)", "check_predictor.py"):
        assert needle in text
    assert "sample_submission.h5" not in text and "floor" not in text


def test_make_submission_points_predictor_challenges_at_the_predictor(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "submission" / "predictor").mkdir(parents=True)
    metadata = tmp_path / "metadata.json"
    run = {"competition_id": GIFT, "run_dir": str(run_dir), "success": True}
    metadata.write_text(json.dumps({"runs": {"r": run}}))
    output = tmp_path / "submission.jsonl"
    args = ["--metadata", metadata, "--output", output, "--data-dir", tmp_path / "data"]
    subprocess.run(
        [sys.executable, "experiments/make_submission.py", *map(str, args)],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        check=True,
        capture_output=True,
    )
    (line,) = map(json.loads, output.read_text().splitlines())
    assert line["submission_path"] == str(run_dir / "submission" / "predictor")


def _imports_outside_try_import_error(tree) -> list[str]:
    """Modules imported other than in `try: ... except ImportError:`."""
    import ast

    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            isinstance(h.type, ast.Name) and h.type.id == "ImportError" for h in node.handlers
        ):
            guarded |= {id(n) for b in node.body for n in ast.walk(b)}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and id(node) not in guarded:
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and id(node) not in guarded and node.level == 0:
            out.append(node.module)
    return out


def test_the_local_check_gets_the_metric_with_its_modules(tmp_path):
    from timeseriesgym.predictor import prepare

    grade = tmp_path / "grade.py"
    grade.write_text("import numpy as np\nfrom .mase import mase\nfrom . import other\n")
    assert prepare.metric_files(grade) == [grade, tmp_path / "mase.py", tmp_path / "other.py"]
    for cid in (GIFT, TSL):
        files = prepare.metric_files(COMPETITIONS / cid / "grade.py")
        assert files[0].name == "grade.py" and all(f.is_file() for f in files)


def test_the_public_tools_are_standalone():
    """The modules copied into the public data import nothing from timeseriesgym (but each
    other), so the local check runs with the agent's Python."""
    import ast

    from timeseriesgym.predictor import prepare

    package = REPO / "timeseriesgym" / "predictor"
    for name in prepare.TOOLS:
        imported = _imports_outside_try_import_error(ast.parse((package / name).read_text()))
        assert not [m for m in imported if m.split(".")[0] == "timeseriesgym"], name
