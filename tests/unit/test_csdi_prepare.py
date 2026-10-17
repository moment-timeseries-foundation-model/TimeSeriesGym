"""Tests for the data of the CSDI PM2.5 challenges (csdi-reproduce-pm25, csdi-implement-model-pm25).

The agent gets CSDI's data with the test months' evaluation targets withheld; the grader gets
their true values.
"""

import hashlib
import importlib
import io
import json
import pickle
import zipfile

import numpy as np
import pandas as pd
import pytest

from timeseriesgym.registry import registry

STATIONS = ["001001", "001002", "001003"]
# (time, ground, missing): May, July and January are training months; June, December and
# March are test months.
ROWS = [
    ("2014/05/31 23:00:00", ["10", "20", ""], ["10", "", ""]),
    ("2014/06/01 00:00:00", ["11", "21", "31"], ["11", "", ""]),
    ("2014/06/01 01:00:00", ["12", "", "32"], ["", "", "32"]),
    ("2014/07/01 00:00:00", ["13", "23", "33"], ["", "23", "33"]),
    ("2014/12/31 23:00:00", ["14", "24", "34"], ["14", "24", ""]),
    ("2015/01/01 00:00:00", ["15", "25", "35"], ["15", "25", "35"]),
    ("2015/03/01 00:00:00", ["", "26", "36"], ["", "26", ""]),
]
# The entries observed in ground and missing in `missing`, in the test months.
TARGETS = [
    ("2014/06/01 00:00:00_001002", 21.0),
    ("2014/06/01 00:00:00_001003", 31.0),
    ("2014/06/01 01:00:00_001001", 12.0),
    ("2014/12/31 23:00:00_001003", 34.0),
    ("2015/03/01 00:00:00_001003", 36.0),
]


def _module(competition_id="csdi-reproduce-pm25"):
    return importlib.import_module(f"timeseriesgym.competitions.{competition_id}.prepare")


def _table(rows, column, header=None, line_end="\r\n"):
    lines = [header or ["datetime", *STATIONS]] + [[row[0], *row[column]] for row in rows]
    return "".join(",".join(line) + line_end for line in lines).encode()


def _csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), dtype={"id": str})


def test_split_withholds_exactly_the_test_months_evaluation_targets():
    split = _module().split_pm25(_table(ROWS, 1), _table(ROWS, 2))

    answers = _csv(split["answers"])
    assert list(answers.itertuples(index=False, name=None)) == TARGETS
    assert split["answers"].startswith(b"id,value\n2014/06/01 00:00:00_001002,21.0\n")
    sample = _csv(split["sample_submission"])
    assert sample["id"].tolist() == answers["id"].tolist() and (sample["value"] == 0.0).all()

    # The targets are empty fields, the only change; CRLF line ends and every other byte stay.
    withheld = {id_ for id_, _ in TARGETS}
    expected = [
        (
            time,
            ["" if f"{time}_{s}" in withheld else v for s, v in zip(STATIONS, ground, strict=True)],
            None,
        )
        for time, ground, _ in ROWS
    ]
    assert split["ground"] == _table(expected, 1)
    public = pd.read_csv(io.BytesIO(split["ground"]), index_col="datetime", dtype=str)
    for id_, _ in TARGETS:
        time, station = id_.rsplit("_", 1)
        assert pd.isna(public.loc[time, station])  # no true value at any target position


def test_meanstd_is_csdis_normalizer_of_the_training_months():
    split = _module().split_pm25(_table(ROWS, 1), _table(ROWS, 2))
    mean, std = pickle.loads(split["meanstd"])

    # As CSDI's download.py computes pm25_meanstd.pk.
    df = pd.read_csv(io.BytesIO(_table(ROWS, 1)), index_col="datetime", parse_dates=True)
    df = df[~df.index.month.isin([3, 6, 9, 12])]
    for array, row in ((mean, "mean"), (std, "std")):
        assert isinstance(array, np.ndarray) and array.dtype == np.float64
        np.testing.assert_allclose(array, df.describe().loc[row].values, rtol=1e-15)
    np.testing.assert_array_equal(mean, [38 / 3, 68 / 3, 34.0])  # May, July and January
    # The pickle names only `numpy.array`, not numpy's internal modules that differ by version.
    assert b"numpy" in split["meanstd"] and b"core" not in split["meanstd"]


def test_split_is_deterministic():
    module = _module()
    assert module.split_pm25(_table(ROWS, 1), _table(ROWS, 2)) == module.split_pm25(
        _table(ROWS, 1), _table(ROWS, 2)
    )


@pytest.mark.parametrize(
    "ground,missing,match",
    [
        (_table(ROWS, 1), _table(ROWS[:1], 2), "same stations and times"),
        (_table(ROWS, 1), _table(ROWS, 2, header=["datetime", "a", "b", "c"]), "same stations"),
        # A value in `missing` must be that of ground: CSDI's mask is ground's minus missing's.
        (_table(ROWS[:1], 1), _table([(*ROWS[0][:2], ["10", "20", "5"])], 2), "has '5' where"),
        (_table(ROWS[:1], 1), _table([(*ROWS[0][:2], ["10", "21", ""])], 2), "has '21' where"),
        (
            _table([(ROWS[0][0], ["1.5", "", ""], None)], 1),
            _table([(ROWS[0][0], None, ["", "", ""])], 2),
            "whole number",
        ),
        (_table(ROWS, 1, line_end="\n"), _table(ROWS, 2), "CRLF"),
        (_table(ROWS, 1) + b"x", _table(ROWS, 2), "CRLF"),
        (_table(ROWS, 1) + b"x\r\n", _table(ROWS, 2), "different lengths"),
    ],
    ids=["times", "stations", "extra", "other", "decimal", "lf", "end", "row"],
)
def test_split_rejects_files_it_does_not_expect(ground, missing, match):
    with pytest.raises(ValueError, match=match):
        _module().split_pm25(ground, missing)


def test_fetch_stmvl_downloads_once_and_checks_the_sha256(tmp_path, monkeypatch):
    module = _module()
    payload = b"zip bytes"
    monkeypatch.setattr(module, "STMVL_SHA256", hashlib.sha256(payload).hexdigest())
    downloads = []

    def fake_download(url, destination):
        downloads.append(url)
        destination.write_bytes(payload)
        return destination

    monkeypatch.setattr(module, "download_file", fake_download)
    raw = tmp_path / "raw"
    assert module.fetch_stmvl(raw).read_bytes() == payload
    assert module.fetch_stmvl(raw) == raw / "STMVL-Release.zip"  # cached: no second download
    assert downloads == [module.STMVL_URL]

    (raw / "STMVL-Release.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        module.fetch_stmvl(raw)

    monkeypatch.setattr(module, "STMVL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        module.fetch_stmvl(tmp_path / "fresh")
    assert list((tmp_path / "fresh").iterdir()) == []  # a bad download is not kept


def test_strip_test_values_removes_the_repositorys_saved_test_targets(tmp_path):
    csdi = tmp_path / "CSDI"
    run = csdi / "save" / "pm25_validationindex0_20250513_210308"
    run.mkdir(parents=True)
    for name in ("generated_outputs_nsample10.pk", "result_nsample10.pk", "model.pth"):
        (run / name).write_bytes(b"x")
    figure = [{"output_type": "display_data", "data": {"image/png": "iVBOR"}}]
    cells = [
        {"cell_type": "code", "source": ["###healthcare###\n"], "outputs": figure},
        {"cell_type": "code", "source": ["###airquality###\n", "k = 10\n"], "outputs": figure},
    ]
    (csdi / "visualize_examples.ipynb").write_text(json.dumps({"cells": cells}, indent=1) + "\n")

    _module().strip_test_values(csdi)
    assert sorted(p.name for p in run.iterdir()) == ["model.pth", "result_nsample10.pk"]
    stripped = json.loads((csdi / "visualize_examples.ipynb").read_text())["cells"]
    assert stripped[0]["outputs"] == figure and stripped[1]["outputs"] == []


@pytest.mark.parametrize(
    "competition_id,branch,commit",
    [
        ("csdi-reproduce-pm25", "easy", "76a4cb7e2b4b5c0bb5eaba04b22e45b67a930091"),
        ("csdi-implement-model-pm25", "pm25", "80915dfa3e6a83f0cbd43696bd55124a6257999a"),
    ],
)
def test_prepare_writes_csdis_layout_and_the_configured_files(
    tmp_path, monkeypatch, competition_id, branch, commit
):
    shared = _module()
    competition = registry.set_data_dir(tmp_path).get_competition(competition_id)
    raw, public, private = competition.raw_dir, competition.public_dir, competition.private_dir
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "STMVL-Release.zip", "w") as archive:
        for name, data in [
            ("pm25_ground.txt", _table(ROWS, 1)),
            ("pm25_missing.txt", _table(ROWS, 2)),
            ("pm25_latlng.txt", b"sensor_id,latitude,longitude\r\n"),
        ]:
            archive.writestr(f"Code/STMVL/SampleData/{name}", data)
    zip_hash = hashlib.sha256((raw / "STMVL-Release.zip").read_bytes()).hexdigest()
    monkeypatch.setattr(shared, "STMVL_SHA256", zip_hash)

    def no_download(url, destination):
        raise AssertionError("downloaded, although the zip is in `raw`")

    monkeypatch.setattr(shared, "download_file", no_download)
    clones = []

    def fake_clone(url, destination, branch=None, commit=None):
        clones.append((url, branch, commit))
        (destination / "save" / "run").mkdir(parents=True)
        (destination / "save" / "run" / "generated_outputs_nsample10.pk").write_bytes(b"x")
        (destination / "dataset_pm25.py").write_text("# CSDI\n")
        return destination

    monkeypatch.setattr(shared, "clone_repository", fake_clone)
    competition.prepare_fn(raw=raw, public=public, private=private)

    assert clones == [(shared.CSDI_URL, branch, commit)]
    files = sorted(str(p.relative_to(public)) for p in public.rglob("*") if p.is_file())
    assert files == [
        "CSDI/data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
        "CSDI/data/pm25/Code/STMVL/SampleData/pm25_latlng.txt",
        "CSDI/data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
        "CSDI/data/pm25/pm25_meanstd.pk",
        "CSDI/dataset_pm25.py",
        "sample_submission.csv",
    ]
    assert [p.name for p in private.iterdir()] == ["answers.csv"]
    assert competition.answers == private / "answers.csv"
    assert competition.sample_submission == public / "sample_submission.csv"
    split = shared.split_pm25(_table(ROWS, 1), _table(ROWS, 2))
    sample_data = public / "CSDI" / "data" / "pm25" / "Code" / "STMVL" / "SampleData"
    assert (sample_data / "pm25_ground.txt").read_bytes() == split["ground"]
    assert (sample_data / "pm25_missing.txt").read_bytes() == _table(ROWS, 2)
    assert competition.answers.read_bytes() == split["answers"]
    assert competition.sample_submission.read_bytes() == split["sample_submission"]
