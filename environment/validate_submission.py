"""The agent's format check: `validate_submission [file or folder]` (default: /home/submission).

The agent's container has neither the test answers nor the graders: each run is graded in a
separate container after the agent's has stopped (agents/grading.py). This check compares a
submission with the challenge's public sample submission only, so it never gives a score. A
submission it reports as valid can still be rejected when it is graded for values its metric
cannot use, such as negative predictions for an RMSLE metric.

What it checks is a contract derived from the sample submission on the host
(`describe_sample`, called by agents/validation.py) and installed in each agent's container at
CONTRACT_PATH:

- CSV: the sample's columns and number of rows; the values of its id columns (the columns
  whose values are not all the same in the sample) in the same order; values of the type of the
  sample's, without missing or infinite numbers.
- .npy: the sample's shape, numbers only, finite. .h5: the datasets under `submission`, with the
  sample's shapes (and fields), numbers only, finite, stored in the file itself.
- .pkl: the sample's structure (keys, lengths, kinds of values), plain data only.
- .py: valid Python; missing classes, functions and arguments the grader runs are noted.
- Folders (hyperparameter search, feature enhancement): the files their grader reads.
- Predictors (`/home/submission/predictor/`): the grader's own format check of the directory,
  `predictor_tools/contract.py` in the public data, which parses predict.py and never runs it.

Standalone (the standard library, and numpy, pandas or h5py for those formats): it runs in the
agent's image, which has no TimeSeriesGym code.
"""

import argparse
import ast
import json
import numbers
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

CONTRACT_PATH = Path("/opt/timeseriesgym/contract.json")
VALID = "Submission is valid."
INVALID = "Submission invalid!"
MAX_STRUCTURE_NODES = 100_000  # of a pickled sample described in the contract
MAX_H5_OBJECTS = 1_000_000
MAX_H5_DEPTH = 32
CHUNK_ELEMENTS = 1 << 24  # finite checks read arrays in chunks of about this many elements


class FormatError(Exception):
    """The submission does not have the format of the sample submission."""


# ---------------------------------------------------------------- describing the sample (host)


def describe_sample(sample: Path, **options) -> dict:
    """The format of a sample submission file. Options (CSV only, from the competition's
    `submission_format` config): `fixed_rows` (default true; false for free-form rows such as
    detected events) and `id_columns` (instead of the columns that vary in the sample)."""
    describers = {
        ".csv": _describe_csv,
        ".npy": _describe_npy,
        ".h5": _describe_h5,
        ".hdf5": _describe_h5,
        ".pkl": _describe_pkl,
    }
    if sample.suffix not in describers:
        raise ValueError(f"{sample.name}: no format check for {sample.suffix!r} files")
    if options and sample.suffix != ".csv":
        raise ValueError(f"{sample.name}: options {sorted(options)} apply to CSV files only")
    return {"suffix": sample.suffix, **describers[sample.suffix](sample, **options)}


def _describe_csv(sample: Path, fixed_rows: bool = True, id_columns: list | None = None) -> dict:
    import pandas as pd

    ref = pd.read_csv(sample, float_precision="round_trip")
    if id_columns is None:
        id_columns = [c for c in ref.columns if fixed_rows and ref[c].nunique(dropna=False) > 1]
    unknown = [c for c in id_columns if c not in ref.columns]
    if unknown:
        raise ValueError(f"{sample.name}: id columns {unknown} are not in the sample")
    values = {}
    for column in ref.columns:
        if column in id_columns:
            continue
        series = ref[column]
        if pd.api.types.is_bool_dtype(series):
            kind = "bool"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "number"
        else:
            kind = "text"
        values[column] = {"type": kind, "missing": bool(series.isna().any())}
    return {
        "format": "csv",
        "columns": [str(c) for c in ref.columns],
        "rows": len(ref) if fixed_rows else None,
        "id_columns": list(id_columns),
        "value_columns": values,
    }


def _describe_npy(sample: Path) -> dict:
    import numpy as np

    array = np.load(sample, mmap_mode="r", allow_pickle=False)
    return {"format": "npy", "shape": list(array.shape)}


def _describe_h5(sample: Path) -> dict:
    import h5py

    with h5py.File(sample, "r") as f:
        datasets = _h5_datasets(f)
        return {
            "format": "h5",
            "datasets": {name: _h5_shape(ds) for name, ds in sorted(datasets.items())},
        }


def _describe_pkl(sample: Path) -> dict:
    with open(sample, "rb") as f:
        return {"format": "pkl", "structure": _structure(_load_plain_pickle(f))}


# ---------------------------------------------------------------- checking a submission (agent)


def check(target: Path, contract: dict) -> tuple[Path, list[str]]:
    """Checks the submission at `target` (a file, or the folder it is in); returns the path
    that was checked and notes on it. Raises FormatError if it does not have the format."""
    kind = contract["kind"]
    if kind == "file":
        path, notes = _pick_file(target, contract)
        return path, notes + _check_file(path, contract["file"], contract)
    if kind == "folder":
        return _check_folder(target, contract)
    if kind == "predictor":
        return _check_predictor(target, contract)
    raise ValueError(f"unknown contract kind {kind!r}")


def _pick_file(target: Path, contract: dict) -> tuple[Path, list[str]]:
    """The file that is graded, as experiments/make_submission.py picks it: of the files in the
    folder with the sample's suffix, the most recently written."""
    suffix = contract["file"]["suffix"]
    folder = target if target.is_dir() else target.parent
    candidates = []
    if folder.is_dir():
        candidates = [p for p in folder.iterdir() if p.is_file() and p.suffix == suffix]
    last = max(candidates, key=lambda p: (p.stat().st_mtime, p.name)) if candidates else None
    notes = []
    if target.is_dir():
        if last is None:
            raise FormatError(f"{target} has no {suffix} file")
        if len(candidates) > 1:
            notes.append(
                f"{target} has {len(candidates)} {suffix} files; only the most recently "
                f"written, {last.name}, is graded"
            )
        target = last
    elif not target.is_file():
        raise FormatError(f"{target} does not exist")
    elif target.suffix != suffix:
        raise FormatError(
            f"{target.name} is not a {suffix} file, as the sample submission "
            f"{contract['file']['sample']} is"
        )
    if not _inside(target, Path(contract["submission_dir"])):
        notes.append(f"only files in {contract['submission_dir']} are graded")
    elif last is not None and last != target:
        notes.append(f"{last.name} was written later in {folder}: only that file is graded")
    return target, notes


def _inside(path: Path, folder: Path) -> bool:
    try:  # resolved: scaffolds link their own folders to it (e.g. AIDE's best_submission)
        return path.resolve().parent == folder.resolve()
    except OSError:
        return False


def _check_file(path: Path, c: dict, contract: dict) -> list[str]:
    checks = {
        "csv": _check_csv,
        "npy": _check_npy,
        "h5": _check_h5,
        "pkl": _check_pkl,
        "py": _check_py,
        "any": lambda path, c, contract: [],
    }
    try:
        return checks[c["format"]](path, c, contract) or []
    except (FormatError, MemoryError):
        raise
    except Exception as e:  # whatever the file makes a reader raise is a format error
        raise FormatError(f"cannot check {path.name}: {type(e).__name__}: {e}") from None


def _check_folder(target: Path, contract: dict) -> tuple[Path, list[str]]:
    """Hyperparameter search and feature enhancement: the files their graders read, in the
    submission folder; the solution code in the code folder."""
    files = contract["files"]
    if target.is_file():  # one of the folder's files, or a file of its prediction files' format
        notes = [f"pass the folder {target.parent} to check all of its files"]
        if target.name in files:
            return target, notes + _check_file(target, files[target.name], contract)
        predictions = [name for name, c in files.items() if c["format"] != "any"]
        formats = {json.dumps(files[name], sort_keys=True) for name in predictions}
        if len(formats) != 1 or target.suffix != files[predictions[0]]["suffix"]:
            raise FormatError(
                f"{target.name} is not one of the files of the submission folder "
                f"({', '.join(files)}); check the folder itself"
            )
        notes.append(f"checked as a prediction file ({', '.join(predictions)})")
        return target, notes + _check_file(target, files[predictions[0]], contract)
    if not target.is_dir():
        raise FormatError(f"{target} is not a folder")
    missing = [name for name in files if not (target / name).is_file()]
    if missing:
        raise FormatError(f"{target} has no {', '.join(missing)}")
    notes = []
    for name, c in files.items():
        notes += [f"{name}: {note}" for note in _check_file(target / name, c, contract)]
    code_dir = Path(contract["code_dir"])
    if contract.get("solution_code") and not any(code_dir.rglob("*.py")):
        raise FormatError(
            f"{code_dir} has no .py file: the code of the search is graded with the submission"
        )
    project = contract.get("project_files", [])
    if project:  # read from the submission folder if it has the first, else from the code folder
        where = target if (target / project[0]).is_file() else code_dir
        missing = [name for name in project if not (where / name).is_file()]
        if missing:
            raise FormatError(f"{where} has no {', '.join(missing)}")
        for name in contract.get("optional_project_files", []):
            if not (where / name).is_file():
                notes.append(f"{where / name} does not exist; it counts toward the score")
    return target, notes


def _check_predictor(target: Path, contract: dict) -> tuple[Path, list[str]]:
    """Predictor challenges: the folder `predictor` of the submission folder (or another folder,
    or its predict.py alone), with the grader's format check, which never imports or runs it."""
    c = contract["predictor"]
    graded = Path(contract["submission_dir"]) / c["dir"]
    if target.is_dir() and _same_path(target, Path(contract["submission_dir"])):
        target = graded
    rules = _load_tool(Path(c["tools"]) / "contract.py")
    notes = []
    try:
        if target.is_dir():
            rules.check_predictor(target, int(c["max_bytes"]), int(c["max_files"]))
        elif target.is_file() and target.suffix == ".py":
            rules.check_entry(target)
            notes.append(f"pass the folder {target.parent} to check all of its files")
        elif target.is_file():
            raise FormatError(
                f"this challenge takes a predictor, the folder {graded} with predict.py (see "
                "description.md), not a file of predictions"
            )
        else:
            raise FormatError(f"{target} does not exist")
    except (rules.ContractError, OSError) as e:
        raise FormatError(str(e)) from None
    folder = target if target.is_dir() else target.parent
    if not _same_path(folder, graded):
        notes.append(f"only {graded} is graded")
    notes.append(
        f"it was not run: `python {contract['data_dir']}/check_predictor.py` runs it as the "
        "grader does, on validation windows"
    )
    return target, notes


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _load_tool(path: Path):
    """A module of the challenge's public tools (standard library only), from the read-only
    public data."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_tsgym_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_csv(path: Path, c: dict, contract: dict) -> list[str]:
    import pandas as pd

    try:
        sub = pd.read_csv(path, float_precision="round_trip")
    except Exception as e:
        raise FormatError(f"cannot read {path.name} as CSV: {type(e).__name__}: {e}") from None
    columns, want = [str(x) for x in sub.columns], c["columns"]
    missing = [x for x in want if x not in columns]
    extra = [x for x in columns if x not in want]
    if missing or extra:
        details = [f"missing {missing}" if missing else "", f"unexpected {extra}" if extra else ""]
        raise FormatError(
            f"the columns must be the sample submission's, {want} "
            f"({', '.join(d for d in details if d)})"
        )
    sub.columns = columns
    notes = []
    if columns != want:
        notes.append(f"the columns are not in the sample submission's order, {want}")
    if c["rows"] is not None and len(sub) != c["rows"]:
        raise FormatError(f"{len(sub)} rows; the sample submission has {c['rows']}")
    for column in c["id_columns"]:
        if sub[column].isna().any():
            raise FormatError(f"column {column!r} has missing values")
    if c["rows"] is not None and c["id_columns"]:
        sample = Path(contract["data_dir"]) / c["sample"]
        ref = pd.read_csv(sample, usecols=c["id_columns"], float_precision="round_trip")
        _check_ids(sub, ref, c["id_columns"])
    for column, spec in c["value_columns"].items():
        _check_values(sub[column], column, spec)
    return notes


def _check_ids(sub, ref, columns: list) -> None:
    if all(_same(sub[col], ref[col]) for col in columns):
        return
    sub_keys, ref_keys = _keys(sub, ref, columns), _keys(ref, sub, columns)
    by = columns[0] if len(columns) == 1 else columns
    if Counter(sub_keys) == Counter(ref_keys):
        raise FormatError(
            f"the rows are not in the sample submission's order of {by!r}: write them in it "
            "(some graders match rows by position)"
        )
    missing = list((Counter(ref_keys) - Counter(sub_keys)).elements())
    unknown = list((Counter(sub_keys) - Counter(ref_keys)).elements())
    raise FormatError(
        f"the values of {by!r} must be the sample submission's: {len(missing)} of them are "
        f"missing{_examples(missing)}, {len(unknown)} are not in it{_examples(unknown)}"
    )


def _numeric(series) -> bool:
    import pandas as pd

    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _same(a, b) -> bool:
    import numpy as np

    if len(a) != len(b):
        return False
    if _numeric(a) and _numeric(b):
        return bool(np.array_equal(a.to_numpy(), b.to_numpy()))
    return a.astype(str).reset_index(drop=True).equals(b.astype(str).reset_index(drop=True))


def _keys(df, other, columns: list) -> list[tuple]:
    """Rows as comparable tuples: numbers as floats if both sides are numeric, else text."""
    parts = []
    for col in columns:
        if _numeric(df[col]) and _numeric(other[col]):
            parts.append(df[col].astype(float).tolist())
        else:
            parts.append(df[col].astype(str).tolist())
    return list(zip(*parts, strict=True))


def _examples(keys: list) -> str:
    if not keys:
        return ""
    shown = [k[0] if len(k) == 1 else k for k in keys[:3]]
    return f" (e.g. {', '.join(repr(k) for k in shown)})"


def _check_values(series, column: str, spec: dict) -> None:
    import numpy as np
    import pandas as pd

    missing = int(series.isna().sum())
    if missing and not spec["missing"]:
        raise FormatError(f"column {column!r} has {missing} missing values")
    if spec["type"] == "bool" and not pd.api.types.is_bool_dtype(series.dropna()):
        raise FormatError(f"column {column!r} must hold True or False, as the sample's does")
    if spec["type"] == "number":
        if not _numeric(series):
            raise FormatError(f"column {column!r} must hold numbers, as the sample's does")
        if not np.isfinite(series.dropna().to_numpy(dtype=float)).all():
            raise FormatError(f"column {column!r} has infinite values")


def _check_numbers(dtype, name: str) -> None:
    if dtype.kind not in "biuf" or dtype.itemsize > 8:
        raise FormatError(f"{name} must hold numbers (up to 64-bit), not {dtype}")


def _check_finite(array, name: str) -> None:
    """Chunk by chunk along the first axis (arrays may be memory-mapped or HDF5 datasets)."""
    import numpy as np

    if array.dtype.kind != "f" or array.size == 0:
        return
    if array.ndim == 0:
        chunks = [array[()]]
    else:
        step = max(1, CHUNK_ELEMENTS // max(1, array.size // array.shape[0]))
        chunks = (array[i : i + step] for i in range(0, array.shape[0], step))
    for chunk in chunks:
        if not np.isfinite(chunk).all():
            raise FormatError(f"{name} has NaN or infinite values")


def _check_npy(path: Path, c: dict, contract: dict) -> None:
    import numpy as np

    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as e:
        raise FormatError(f"cannot read {path.name} as a .npy array: {e}") from None
    if not isinstance(array, np.ndarray):
        raise FormatError(f"{path.name} is not a single .npy array")
    if list(array.shape) != c["shape"]:
        raise FormatError(
            f"{path.name} has shape {tuple(array.shape)}; the sample submission's is "
            f"{tuple(c['shape'])}"
        )
    _check_numbers(array.dtype, path.name)
    _check_finite(array, path.name)


def _h5_datasets(f) -> dict:
    """The datasets under `submission`, by name, walking the links without following them:
    only hard links to groups and to datasets stored in the file itself, as the grader reads."""
    import h5py

    if "submission" not in f:
        raise FormatError("the file must hold a dataset or group named 'submission'")
    stack, found, count = [(f, "", 0)], {}, 0
    while stack:
        group, prefix, depth = stack.pop()
        if depth > MAX_H5_DEPTH:
            raise FormatError(f"groups are nested deeper than {MAX_H5_DEPTH} levels")
        for key in group:
            name = f"{prefix}/{key}" if prefix else key
            count += 1
            if count > MAX_H5_OBJECTS:
                raise FormatError(f"more than {MAX_H5_OBJECTS} HDF5 objects")
            link = group.get(key, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise FormatError(f"{name} is an HDF5 {type(link).__name__}; store the data")
            obj = group.get(key)
            if isinstance(obj, h5py.Group):
                stack.append((obj, name, depth + 1))
            elif not isinstance(obj, h5py.Dataset):
                raise FormatError(f"{name}: only HDF5 groups and datasets are allowed")
            elif obj.is_virtual or obj.external:
                raise FormatError(f"{name}: virtual or externally stored datasets are not allowed")
            elif name == "submission" or name.startswith("submission/"):
                found[name] = obj
    return found


def _h5_shape(ds) -> dict:
    names = ds.dtype.names
    fields = {n: list(ds.dtype[n].shape) for n in names} if names else None
    return {"shape": list(ds.shape), "fields": fields}


def _check_h5(path: Path, c: dict, contract: dict) -> None:
    import h5py

    try:
        f = h5py.File(path, "r")
    except Exception as e:
        raise FormatError(f"cannot open {path.name} as HDF5: {e}") from None
    with f:
        found, want = _h5_datasets(f), c["datasets"]
        missing = sorted(set(want) - set(found))
        extra = sorted(set(found) - set(want))
        if missing or extra:
            raise FormatError(
                f"the datasets must be the sample submission's ({len(want)}): "
                f"{len(missing)} missing{_examples([(m,) for m in missing])}, "
                f"{len(extra)} unexpected{_examples([(x,) for x in extra])}"
            )
        for name, spec in want.items():
            ds, got = found[name], _h5_shape(found[name])
            if got != spec:
                raise FormatError(
                    f"{name} has shape {got['shape']} and fields {got['fields']}; the sample "
                    f"submission's has {spec['shape']} and {spec['fields']}"
                )
            for field in ds.dtype.names or [None]:
                dtype = ds.dtype[field].base if field else ds.dtype
                label = f"{name}[{field!r}]" if field else name
                _check_numbers(dtype, label)
                if field is None:
                    _check_finite(ds, label)
                elif dtype.kind == "f":
                    _check_finite(ds.fields(field)[()], label)


# The globals the grader's unpickler allows (timeseriesgym/utils.py, _PLAIN_PICKLE_GLOBALS).
PLAIN_PICKLE_GLOBALS = {
    *(("builtins", name) for name in ("list", "dict", "tuple", "set", "frozenset", "complex")),
    ("collections", "defaultdict"),
    ("collections", "OrderedDict"),
    ("_codecs", "encode"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
}


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__builtin__":
            module = "builtins"
        if (module, name) not in PLAIN_PICKLE_GLOBALS:
            raise FormatError(
                f"the pickle references `{module}.{name}`; submissions may only contain plain "
                "containers, strings and numbers (numpy scalars included)"
            )
        return super().find_class(module, name)


def _load_plain_pickle(file):
    return _PlainUnpickler(file).load()


def _structure(obj, budget: list | None = None):
    """A JSON description of plain data: its containers, keys and lengths, and the kinds of
    its values. Beyond MAX_STRUCTURE_NODES nodes, "any"."""
    budget = [MAX_STRUCTURE_NODES] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        return "any"
    if isinstance(obj, bool) or getattr(getattr(obj, "dtype", None), "kind", None) == "b":
        return "bool"
    if isinstance(obj, numbers.Number):
        return "number"
    if isinstance(obj, str | bytes):
        return type(obj).__name__
    if obj is None:
        return "none"
    if isinstance(obj, dict):
        if not all(isinstance(k, str | int | float | bool) or k is None for k in obj):
            return "any"
        return {"dict": [[k, _structure(v, budget)] for k, v in obj.items()]}
    if isinstance(obj, list | tuple):
        return {type(obj).__name__: [_structure(v, budget) for v in obj]}
    if isinstance(obj, set | frozenset):
        return {"set": len(obj)}
    return "any"


def _compare_structure(got, want, where: str) -> None:
    if want == "any":
        return
    mismatch = f"{where} is {_kind(got)}; in the sample submission it is {_kind(want)}"
    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise FormatError(mismatch)
        return
    (kind,), (got_kind,) = want.keys(), got.keys()
    if kind != got_kind:
        raise FormatError(mismatch)
    if kind == "set":
        if got["set"] != want["set"]:
            raise FormatError(f"{where} has {got['set']} items; the sample's has {want['set']}")
        return
    if kind == "dict":
        got_items = dict(got["dict"])
        want_keys = [k for k, _ in want["dict"]]
        missing = [k for k in want_keys if k not in got_items]
        extra = [k for k in got_items if k not in set(want_keys)]
        if missing or extra:
            raise FormatError(
                f"{where} must have the sample submission's keys: {len(missing)} missing"
                f"{_examples([(k,) for k in missing])}, {len(extra)} unexpected"
                f"{_examples([(k,) for k in extra])}"
            )
        for key, value in want["dict"]:
            _compare_structure(got_items[key], value, f"{where}[{key!r}]")
        return
    if len(got[kind]) != len(want[kind]):
        raise FormatError(
            f"{where} has {len(got[kind])} items; in the sample submission it has "
            f"{len(want[kind])}"
        )
    for i, (g, w) in enumerate(zip(got[kind], want[kind], strict=True)):
        _compare_structure(g, w, f"{where}[{i}]")


def _kind(structure) -> str:
    if isinstance(structure, str):
        return {"none": "None", "any": "too large to check"}.get(structure, f"a {structure}")
    return f"a {next(iter(structure))}"


def _check_pkl(path: Path, c: dict, contract: dict) -> None:
    try:
        with open(path, "rb") as f:
            obj = _load_plain_pickle(f)
    except FormatError:
        raise
    except Exception as e:
        raise FormatError(f"cannot unpickle {path.name}: {type(e).__name__}: {e}") from None
    _compare_structure(_structure(obj), c["structure"], "the submission")


def _check_py(path: Path, c: dict, contract: dict) -> list[str]:
    try:
        tree = ast.parse(path.read_bytes(), filename=path.name)
    except Exception as e:  # SyntaxError, ValueError (null bytes), RecursionError, ...
        raise FormatError(f"{path.name} is not valid Python: {e}") from None
    defined = {}  # top-level names: their def or class, or None (assigned or imported)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined[node.name] = node
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                defined[(alias.asname or alias.name).split(".")[0]] = None
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        defined[name.id] = None
    notes = []
    for cls, methods in c.get("classes", {}).items():
        if cls not in defined:
            notes.append(f"class {cls} is not defined")
            continue
        if not isinstance(defined[cls], ast.ClassDef):
            continue
        body = {
            n.name: n
            for n in defined[cls].body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for method, spec in methods.items():
            if method not in body:
                notes.append(f"{cls}.{method} is not defined in the class")
            else:
                notes += _argument_notes(body[method], spec, f"{cls}.{method}", method=True)
    for function, spec in c.get("functions", {}).items():
        if function not in defined:
            notes.append(f"function {function} is not defined")
        elif isinstance(defined[function], ast.FunctionDef | ast.AsyncFunctionDef):
            notes += _argument_notes(defined[function], spec, function, method=False)
    if notes:
        notes.append("the grader runs and scores each of these")
    return notes


def _argument_notes(node, spec: dict, name: str, method: bool) -> list[str]:
    """The grader's check of a signature (timeseriesgym/grade_code_helper.py)."""
    a = node.args
    args = [x.arg for x in a.posonlyargs + a.args]
    args += [a.vararg.arg] if a.vararg else []
    args += [x.arg for x in a.kwonlyargs] + ([a.kwarg.arg] if a.kwarg else [])
    required = list(spec["args"])
    if method:
        args, required = [x for x in args if x != "self"], [x for x in required if x != "self"]
    if spec.get("exact") and args != required:
        return [f"{name} must take exactly ({', '.join(required)}), in this order"]
    missing = [x for x in required if x not in args]
    return [f"{name} has no argument {', '.join(missing)}"] if missing else []


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_submission",
        description="Check a submission's format against the sample submission (no score).",
    )
    parser.add_argument("path", nargs="?", help="the submission file or folder")
    parser.add_argument("--contract", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    contract_path = args.contract or Path(os.environ.get("TIMESERIESGYM_CONTRACT", CONTRACT_PATH))
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, ValueError) as e:
        print(f"validate_submission: cannot read the challenge's format ({contract_path}): {e}")
        return 2
    if contract["kind"] == "none":
        print(f"The format of submissions is not checked for this challenge: {contract['note']}")
        return 0
    target = Path(args.path or contract["submission_dir"])
    try:
        path, notes = check(target, contract)
    except FormatError as e:
        print(f"{INVALID} {e}")
        return 1
    print(
        f"{VALID} {path} has the format of the sample submission. This checks the format "
        "only: the submission is scored, and the values its metric needs are checked, when the "
        "run is graded."
    )
    for note in notes:
        print(f"Note: {note}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")  # e.g. file names in a C locale
    sys.exit(main())
