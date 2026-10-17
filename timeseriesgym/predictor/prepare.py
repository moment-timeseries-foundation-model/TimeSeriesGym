"""Helpers of the predictor challenges' prepare.py: the public data end before the test labels.

- TSL long horizon: each CSV is cut at the row of TSL's first test label (cut_csv). The public
  copy of TSL's data_loader.py keeps the full files' borders and scaler (patch_tsl_loader).
- GIFT-Eval: each series is cut before its first test window (cut_gift_dataset), in the same
  Arrow layout, with features the agent's `datasets` 2.1.0 reads; the public dataset.py runs
  with the agent's numpy 1.26 (patch_gift_dataset_py).

Both get `sample_predictor/` (zero forecasts), `predictor_datasets.json` (per dataset: shapes
and window counts, no values) and the local check (`check_predictor.py` with the
`predictor_tools/` package: this package's standalone modules, and the challenge's grade.py
with the modules it imports as `predictor_tools/metric/`).
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from timeseriesgym.utils import load_yaml

from . import protocol

PACKAGE = Path(__file__).resolve().parent
# The standalone modules of this package, copied into the public data for the local check.
TOOLS = ("__init__.py", "protocol.py", "runner.py", "windows.py", "contract.py", "launch.py")
TOOLS += ("check.py",)
TOOLS_DIR = "predictor_tools"
SAMPLE_DIR = "sample_predictor"
METADATA = "predictor_datasets.json"
CHECK_SCRIPT = '''\
"""Run a predictor on validation windows of this data, as the grader runs it on the test windows:

    python /home/data/check_predictor.py [/home/submission/predictor] [--help]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True  # this directory may be read-only

from predictor_tools.check import main  # noqa: E402

sys.exit(main())
'''


def predictor_config(competition_dir: Path) -> dict:
    """The challenge's `predictor_config` (config.yaml) over the grader's defaults."""
    from .grading import DEFAULTS

    return {**DEFAULTS, **load_yaml(competition_dir / "config.yaml")["predictor_config"]}


def metric_files(grade_py: Path) -> list[Path]:
    """grade.py and the modules of its directory that it imports relatively (`from .x import`,
    `from . import x`)."""
    names = set()
    for node in ast.walk(ast.parse(grade_py.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                names.add(node.module.split(".")[0])
            else:
                names.update(alias.name for alias in node.names)
    return [grade_py, *(grade_py.with_name(f"{name}.py") for name in sorted(names))]


def _replace(text: str, old: str, new: str, count: int, where: str) -> str:
    if text.count(old) != count:
        raise ValueError(f"{where}: expected {count} occurrence(s) of {old[:60]!r}")
    return text.replace(old, new)


def write_common(public: Path, grade_py: Path, config: dict, meta: dict, test_windows) -> None:
    """sample_predictor/, the local check and predictor_datasets.json (with the call schedule
    of the test windows, as the grader will run it: counts only, no values)."""
    (public / SAMPLE_DIR).mkdir()
    shutil.copy(PACKAGE / "sample_predictor.py", public / SAMPLE_DIR / "predict.py")
    tools = public / TOOLS_DIR
    tools.mkdir()
    for name in TOOLS:
        shutil.copy(PACKAGE / name, tools / name)
    (tools / "metric").mkdir()
    (tools / "metric" / "__init__.py").write_text("")
    for path in metric_files(grade_py):
        shutil.copy(path, tools / "metric" / path.name)
    keys = ("windows", "timeout_sec", "cpus", "max_bytes", "max_files")
    keys += ("max_batch_windows", "max_batch_values")
    tool_config = {k: config[k] for k in keys}
    grader = load_yaml(grade_py.with_name("config.yaml"))["grader"]["name"]
    tool_config["metric"] = config.get("metric") or grader
    (tools / "config.json").write_text(json.dumps(tool_config, indent=2) + "\n")
    (public / "check_predictor.py").write_text(CHECK_SCRIPT)
    meta = {
        **meta,
        "test_rounds": len(protocol.schedule(test_windows)),
        "test_calls": len(
            protocol.batches(
                test_windows, int(config["max_batch_windows"]), int(config["max_batch_values"])
            )
        ),
    }
    (public / METADATA).write_text(json.dumps(meta, indent=2) + "\n")


# --------------------------------------------------------------------------- TSL

TSL_PATCH_NOTE = """\
# TimeSeriesGym: in this environment the CSVs end where TSL's test split begins; the test rows
# are withheld and the grader runs your predictor on them. Dataset_Custom computes its borders
# and fits its scaler on the full files, whose row counts are recorded here, so the training
# and validation splits and the scaling are those of the full data. flag="test" falls back to
# the validation split, because the test rows are not here.
TSG_FULL_ROWS = {rows}
"""
TSL_BORDERS = """\
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
"""
TSL_BORDERS_PATCHED = """\
        if self.set_type == 2 and len(df_raw) < border2s[2]:  # TimeSeriesGym: test rows withheld
            print('TimeSeriesGym: no test rows here; flag="test" gives the validation split')
            border1s[2], border2s[2] = border1s[1], border2s[1]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
"""
TSL_CUSTOM = """\
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
"""
TSL_CUSTOM_PATCHED = """\
        n_rows = TSG_FULL_ROWS.get(os.path.basename(self.data_path), len(df_raw))  # TimeSeriesGym
        num_train = int(n_rows * 0.7)
        num_test = int(n_rows * 0.2)
        num_vali = n_rows - num_train - num_test
        border1s = [0, num_train - self.seq_len, n_rows - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, n_rows]
"""


def cut_csv(src: Path, dst: Path, rows: int) -> int:
    """The header and the first `rows` data rows of `src`, byte for byte; returns the number
    of data rows of `src`."""
    with open(src, "rb") as f:
        lines = f.readlines()
    if len(lines) - 1 < rows:
        raise ValueError(f"{src}: {len(lines) - 1} rows, fewer than {rows}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.writelines(lines[: 1 + rows])
    return len(lines) - 1


def patch_tsl_loader(path: Path, full_rows: dict[str, int]) -> None:
    """TSL's data_loader.py for the cut CSVs: Dataset_Custom's borders and scaler from the full
    files' row counts (`full_rows`, by file name), and flag "test" without test rows gives the
    validation split (ETT and custom loaders)."""
    text = path.read_text()
    text = _replace(text, TSL_BORDERS, TSL_BORDERS_PATCHED, 3, path.name)
    text = _replace(text, TSL_CUSTOM, TSL_CUSTOM_PATCHED, 1, path.name)
    anchor = "warnings.filterwarnings('ignore')\n"
    rows = json.dumps(dict(sorted(full_rows.items())), indent=4).replace("\n}", ",\n}")
    text = _replace(text, anchor, anchor + "\n" + TSL_PATCH_NOTE.format(rows=rows), 1, path.name)
    path.write_text(text)


# --------------------------------------------------------------------------- GIFT-Eval

GIFT_NUMPY_SHIM = """\
import datasets
import numpy as _np

# TimeSeriesGym: the agent environment's datasets==2.1.0 uses the alias np.object, which
# numpy 1.24 removed; without it, Dataset() fails with AttributeError.
if "object" not in vars(_np):
    _np.object = object
"""


def patch_gift_dataset_py(src: Path, dst: Path) -> None:
    """The public dataset.py: the challenge's, plus the np.object alias for datasets 2.1.0."""
    text = _replace(src.read_text(), "import datasets\n", GIFT_NUMPY_SHIM, 1, src.name)
    dst.write_text(text)


def _sequence_lengths(obj):
    """Hugging Face features with an explicit `"length": -1` on every Sequence: datasets 2.1.0
    (the agent environment's) requires the key and fails with KeyError 'length' without it;
    newer versions write the features without it and read them either way."""
    if isinstance(obj, dict):
        out = {k: _sequence_lengths(v) for k, v in obj.items()}
        if out.get("_type") == "Sequence":
            out.setdefault("length", -1)
        return out
    if isinstance(obj, list):
        return [_sequence_lengths(v) for v in obj]
    return obj


def cut_gift_dataset(src_dir: Path, dst_dir: Path, cut: list[int]) -> None:
    """The dataset's Arrow stream with each row's target cut to its first `cut[i]` values; the
    schema and the other columns unchanged, except Sequence lengths in the features
    (_sequence_lengths)."""
    import pyarrow as pa

    (arrow,) = sorted(src_dir.glob("*.arrow"))
    with pa.memory_map(str(arrow)) as source:
        table = pa.ipc.open_stream(source).read_all()
    target = table.column("target").combine_chunks()
    if len(target) != len(cut):
        raise ValueError(f"{src_dir}: {len(target)} rows, {len(cut)} lengths")
    rows = [target[i].values.slice(0, n) for i, n in enumerate(cut)]
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int32)
    values = pa.concat_arrays(rows) if rows else target.values.slice(0, 0)
    new_target = pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), values)
    new_target = new_target.cast(table.schema.field("target").type)
    i = table.schema.get_field_index("target")
    table = table.set_column(i, table.schema.field("target"), new_target)
    meta = dict(table.schema.metadata or {})
    if b"huggingface" in meta:
        hf = _sequence_lengths(json.loads(meta[b"huggingface"]))
        meta[b"huggingface"] = json.dumps(hf).encode()
        table = table.replace_schema_metadata(meta)
    dst_dir.mkdir(parents=True)
    with pa.OSFile(str(dst_dir / arrow.name), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    info = _sequence_lengths(json.loads((src_dir / "dataset_info.json").read_text()))
    (dst_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    state = json.loads((src_dir / "state.json").read_text())
    state["_fingerprint"] = hashlib.sha256((dst_dir / arrow.name).read_bytes()).hexdigest()[:16]
    (dst_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
