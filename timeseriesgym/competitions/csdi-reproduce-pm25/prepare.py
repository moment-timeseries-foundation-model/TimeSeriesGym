"""Prepares the CSDI PM2.5 challenges (`csdi-implement-model-pm25` uses `prepare_csdi` too).

The agent gets the CSDI repository with the STMVL PM2.5 data in CSDI's own layout
(`CSDI/data/pm25/`), so CSDI's loaders work unchanged. The evaluation targets are the entries
that CSDI's protocol scores: observed in `pm25_ground.txt` but missing in `pm25_missing.txt`,
in the test months. They are removed from the public `pm25_ground.txt`; `answers.csv` holds
their true values. Every output is a deterministic function of the pinned inputs.
"""

import hashlib
import json
import math
import pickle
import zipfile
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

from timeseriesgym.competitions.utils import download_file
from timeseriesgym.utils import clone_repository

CSDI_URL = "https://github.com/moment-timeseries-foundation-model/CSDI.git"
STMVL_URL = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
STMVL_ZIP = "STMVL-Release.zip"
STMVL_SHA256 = "d238fb6a605ef43f96b4428df62ad0ae1e330230f5b1cba62888af7bae3f1700"

# The data files are at this path in the zip and, as CSDI's download.py extracts the zip there,
# under CSDI's `data/pm25/`.
SAMPLE_DATA = "Code/STMVL/SampleData"
GROUND, MISSING, LATLNG = "pm25_ground.txt", "pm25_missing.txt", "pm25_latlng.txt"
MEANSTD = "pm25_meanstd.pk"

# CSDI's dataset_pm25.py tests on these months; download.py leaves them out of pm25_meanstd.pk.
TEST_MONTHS = (3, 6, 9, 12)
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
LINE_END = "\r\n"


def fetch_stmvl(raw: Path) -> Path:
    """The STMVL-Release zip in `raw`, downloaded only if it is not there yet.

    A download is kept only if its SHA-256 is the pinned one; a cached zip is checked as well.
    """
    zip_path = raw / STMVL_ZIP
    if not zip_path.is_file():
        raw.mkdir(parents=True, exist_ok=True)
        part = raw / f"{STMVL_ZIP}.part"
        download_file(STMVL_URL, part)
        try:
            _check_sha256(part)
        except ValueError:
            part.unlink()
            raise
        part.replace(zip_path)
    _check_sha256(zip_path)
    return zip_path


def _check_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != STMVL_SHA256:
        raise ValueError(f"{path} has SHA-256 {digest}, but {STMVL_SHA256} is expected")


def _read_table(text: bytes, name: str) -> tuple[list[str], list[list[str]]]:
    """The header and rows of an STMVL file: comma-separated, CRLF line ends, a missing value is
    an empty field. Raises ValueError unless writing them back gives exactly `text`."""
    *lines, end = text.decode("ascii").split(LINE_END)
    if end or len(lines) < 2:
        raise ValueError(f"{name} is not a table with CRLF line ends")
    header, *rows = (line.split(",") for line in lines)
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"{name} has rows of different lengths")
    if _write_table(header, rows) != text:
        raise ValueError(f"{name} has an unexpected format")
    return header, rows


def _write_table(header: list[str], rows: list[list[str]]) -> bytes:
    return "".join(",".join(line) + LINE_END for line in [header, *rows]).encode("ascii")


def _write_csv(rows: list[tuple[str, float]]) -> bytes:
    return "".join(["id,value\n", *(f"{id_},{value!r}\n" for id_, value in rows)]).encode()


def _mean_std(values: list[int], station: str) -> tuple[float, float]:
    """The mean and the sample standard deviation (as pandas' `describe`), computed exactly and
    rounded once, so that they do not depend on the summation order or the library version."""
    n, total = len(values), sum(values)
    if n < 2:
        raise ValueError(f"Station {station} has fewer than two values in the training months")
    variance = Fraction(n * sum(v * v for v in values) - total * total, n * (n - 1))
    return total / n, math.sqrt(variance)


class _Float64Array:
    """Pickles as `numpy.array(values, "float64")`, so that the file's bytes do not depend on
    the numpy version and it loads with numpy 1 and 2 (CSDI reads it with `pickle.load`)."""

    def __init__(self, values: list[float]):
        self.values = values

    def __reduce__(self):
        return np.array, (self.values, "float64")


def split_pm25(ground: bytes, missing: bytes) -> dict[str, bytes]:
    """Splits STMVL's `pm25_ground.txt` and `pm25_missing.txt` into the challenge files.

    Returns, as bytes:
    - `ground`: `pm25_ground.txt` with every evaluation target made an empty field;
    - `meanstd`: CSDI's `pm25_meanstd.pk`, `[mean, std]` per station over the observed values of
      the months that are not test months;
    - `answers` and `sample_submission`: CSVs with columns `id,value`, one row per evaluation
      target, sorted by `id` (time, then station). `id` is `<datetime>_<station>` as written in
      the file; `value` is the true value, or 0.0 in the sample.
    """
    header, ground_rows = _read_table(ground, GROUND)
    missing_header, missing_rows = _read_table(missing, MISSING)
    if missing_header != header or [r[0] for r in missing_rows] != [r[0] for r in ground_rows]:
        raise ValueError(f"{GROUND} and {MISSING} must have the same stations and times")
    stations = header[1:]

    targets = []
    training_values = [[] for _ in stations]
    public_rows = []
    for ground_row, missing_row in zip(ground_rows, missing_rows, strict=True):
        time = ground_row[0]
        is_test = datetime.strptime(time, TIME_FORMAT).month in TEST_MONTHS
        public_row = list(ground_row)
        for column, station in enumerate(stations, start=1):
            value, kept = ground_row[column], missing_row[column]
            if kept and kept != value:  # else CSDI's evaluation mask is not a 0/1 mask
                raise ValueError(
                    f"{MISSING} has {kept!r} where {GROUND} has {value!r} ({time}, {station})"
                )
            if not value:
                continue
            if not value.isdigit():
                raise ValueError(f"{GROUND} has a value {value!r} that is not a whole number")
            if not is_test:
                training_values[column - 1].append(int(value))
            elif not kept:
                targets.append((f"{time}_{station}", float(value)))
                public_row[column] = ""
        public_rows.append(public_row)
    targets.sort()

    stats = [_mean_std(v, s) for v, s in zip(training_values, stations, strict=True)]
    meanstd = [_Float64Array([m for m, _ in stats]), _Float64Array([s for _, s in stats])]
    return {
        "ground": _write_table(header, public_rows),
        "meanstd": pickle.dumps(meanstd, protocol=4),
        "answers": _write_csv(targets),
        "sample_submission": _write_csv([(id_, 0.0) for id_, _ in targets]),
    }


def strip_test_values(csdi: Path) -> None:
    """Removes the test months' true values that the CSDI repository itself contains: the
    `generated_outputs_nsample*.pk` of a saved run (CSDI's `evaluate` pickles the test targets
    with the samples) and the notebook's air-quality figure, which plots a test window's
    targets."""
    for path in sorted(csdi.rglob("generated_outputs*.pk")):
        path.unlink()
    notebook = csdi / "visualize_examples.ipynb"
    if notebook.is_file():
        cells = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in cells["cells"]:
            if "".join(cell.get("source", [])).startswith("###airquality###"):
                cell["outputs"] = []
        notebook.write_text(json.dumps(cells, indent=1) + "\n", encoding="utf-8")


def write_pm25_data(zip_path: Path, csdi: Path, public: Path, private: Path) -> None:
    """Writes the PM2.5 data into the repository at `csdi`, the sample submission into `public`
    and the answers into `private`."""
    with zipfile.ZipFile(zip_path) as archive:
        files = {name: archive.read(f"{SAMPLE_DATA}/{name}") for name in (GROUND, MISSING, LATLNG)}
    split = split_pm25(files[GROUND], files[MISSING])

    data = csdi / "data" / "pm25"
    (data / SAMPLE_DATA).mkdir(parents=True)
    (data / SAMPLE_DATA / GROUND).write_bytes(split["ground"])
    (data / SAMPLE_DATA / MISSING).write_bytes(files[MISSING])
    (data / SAMPLE_DATA / LATLNG).write_bytes(files[LATLNG])
    (data / MEANSTD).write_bytes(split["meanstd"])
    (public / "sample_submission.csv").write_bytes(split["sample_submission"])
    (private / "answers.csv").write_bytes(split["answers"])


def prepare_csdi(raw: Path, public: Path, private: Path, branch: str, commit: str) -> None:
    zip_path = fetch_stmvl(raw)  # first, so that a failed download stops before the clone
    csdi = clone_repository(CSDI_URL, public / "CSDI", branch=branch, commit=commit)
    strip_test_values(csdi)
    write_pm25_data(zip_path, csdi, public, private)


def prepare(raw: Path, public: Path, private: Path):
    prepare_csdi(
        raw, public, private, branch="easy", commit="76a4cb7e2b4b5c0bb5eaba04b22e45b67a930091"
    )
