"""The PTB-XL challenge zip is reproducible.

The same PTB-XL release gives the same zip, h5 and CSV bytes whatever the directory listing
order, file mtimes, permissions or the time of the build.
"""

import argparse
import ast
import hashlib
import importlib
import os
import time
import zipfile

import h5py
import numpy as np
import pandas as pd
import pytest

from timeseriesgym.utils import ZIP_DATE_TIME, compress, extract

wfdb = pytest.importorskip("wfdb")
download = importlib.import_module(
    "timeseriesgym.competitions.ptb-xl-classification-challenge.download_data"
)


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _reversed_listdir(monkeypatch):
    real_listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path=".": sorted(real_listdir(path), reverse=True))


# ---------------------------------------------------------------- zip


def test_compress_zip_does_not_depend_on_order_mtimes_or_permissions(tmp_path):
    files = {"b.csv": b"label\n0\n", "sub/a.h5": bytes(range(256)) * 64, "run.sh": b"#!/bin/sh\n"}
    zips = []
    builds = [("one", sorted, 1.0e9, 0o600), ("two", reversed, 1.7e9, 0o664)]
    for name, order, mtime, mode in builds:
        src = tmp_path / name
        for rel in order(sorted(files)):  # created in opposite orders
            path = src / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[rel])
            path.chmod(mode | (0o100 if rel == "run.sh" else 0))
            os.utime(path, (mtime, mtime))
        zips.append(tmp_path / f"{name}.zip")
        compress(src, zips[-1])

    assert _md5(zips[0]) == _md5(zips[1])
    with zipfile.ZipFile(zips[0]) as archive:
        infos = archive.infolist()
    assert [info.filename for info in infos] == ["b.csv", "run.sh", "sub/a.h5"]
    assert {info.date_time for info in infos} == {ZIP_DATE_TIME}
    assert {info.compress_type for info in infos} == {zipfile.ZIP_DEFLATED}
    modes = {info.filename: info.external_attr >> 16 for info in infos}
    assert modes == {"b.csv": 0o100644, "run.sh": 0o100755, "sub/a.h5": 0o100644}

    extract(zips[1], tmp_path / "extracted")
    for rel, data in files.items():
        assert (tmp_path / "extracted" / rel).read_bytes() == data


# ---------------------------------------------------------------- record order


def _touch_record(directory, ecg_id):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("hea", "dat"):
        (directory / f"{ecg_id:05d}_lr.{ext}").write_text("")


def test_load_paths_is_sorted_whatever_the_listing_order(tmp_path, monkeypatch):
    for ecg_id in (1002, 3, 1, 1001, 2):
        _touch_record(tmp_path / "records100" / f"{ecg_id // 1000 * 1000:05d}", ecg_id)
    _touch_record(tmp_path / "records500" / "00000", 1)
    (tmp_path / "records100" / "index.html").write_text("")

    records = [("00000", 1), ("00000", 2), ("00000", 3), ("01000", 1001), ("01000", 1002)]
    expected = [str(tmp_path / "records100" / d / f"{ecg_id:05d}_lr") for d, ecg_id in records]
    assert download.load_paths(tmp_path, fs=100) == expected
    _reversed_listdir(monkeypatch)
    assert download.load_paths(tmp_path, fs=100) == expected


# ---------------------------------------------------------------- h5


def test_write_h5_has_no_timestamps_and_the_earliest_format(tmp_path):
    signals = np.arange(2 * 12 * 10, dtype=np.float64).reshape(2, 12, 10)
    labels = np.array([0, 1])

    def timestamped(path):
        with h5py.File(path, "w") as f:
            f.create_dataset("signals", data=signals, track_times=True)

    paths = [tmp_path / f"{i}.h5" for i in range(4)]
    timestamped(paths[0])
    download.write_h5(paths[2], signals=signals, labels=labels)
    time.sleep(1.1)  # the timestamps have a one-second resolution
    timestamped(paths[1])
    download.write_h5(paths[3], signals=signals, labels=labels)

    assert _md5(paths[0]) != _md5(paths[1])  # the sleep is long enough to show timestamps
    assert _md5(paths[2]) == _md5(paths[3])
    assert paths[2].read_bytes()[:9] == b"\x89HDF\r\n\x1a\n\x00"  # superblock version 0
    with h5py.File(paths[2], "r") as f:
        assert list(f) == ["labels", "signals"]
        np.testing.assert_array_equal(f["signals"][:], signals)
        np.testing.assert_array_equal(f["labels"][:], labels)


# ---------------------------------------------------------------- the whole build

CLASSES = {"NORM": "NORM", "IMI": "MI", "STTC": "STTC", "LVH": "HYP", "CRBBB": "CD"}


def _write_release(base):
    """A small PTB-XL-shaped release: 20 records, two per fold, 12 leads of 10 samples."""
    rng = np.random.default_rng(0)
    codes = list(CLASSES)
    rows = []
    for ecg_id in range(1, 21):
        code = codes[ecg_id % len(codes)]
        rows.append(
            {
                "ecg_id": ecg_id,
                "scp_codes": str({code: 100.0}),
                "report": f"report {ecg_id}",
                "strat_fold": (ecg_id - 1) // 2 + 1,
            }
        )
        record_dir = base / "records100" / "00000"
        record_dir.mkdir(parents=True, exist_ok=True)
        wfdb.wrsamp(
            f"{ecg_id:05d}_lr",
            fs=100,
            units=["mV"] * 12,
            sig_name=[f"lead{i}" for i in range(12)],
            p_signal=rng.normal(size=(10, 12)).round(3),
            fmt=["16"] * 12,
            write_dir=str(record_dir),
        )
    pd.DataFrame(rows).to_csv(base / "ptbxl_database.csv", index=False)
    statements = pd.DataFrame(
        {"diagnostic": 1.0, "diagnostic_class": list(CLASSES.values())}, index=codes
    )
    statements.to_csv(base / "scp_statements.csv")


def _build(release, data_dir):
    args = argparse.Namespace(
        fs=100, output_type="single", data_dir=str(data_dir), release_dir=str(release)
    )
    download.main(args)
    return data_dir / "ptb-xl-classification-challenge" / "ptb-xl-classification-challenge.zip"


def test_challenge_zip_is_identical_for_reordered_listings(tmp_path, monkeypatch):
    release = tmp_path / "release"
    _write_release(release)
    first = _build(release, tmp_path / "first")
    with monkeypatch.context() as patch:
        _reversed_listdir(patch)
        second = _build(release, tmp_path / "second")

    assert _md5(first) == _md5(second)
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
    assert names == sorted(
        ["sample_submission.csv", "test_df.h5", "test_label_df.csv", "train_df.h5", "val_df.h5"]
    )

    raw = tmp_path / "raw"
    extract(first, raw)
    database = pd.read_csv(release / "ptbxl_database.csv", index_col="ecg_id")
    encoded = {code: sorted(CLASSES.values()).index(cls) for code, cls in CLASSES.items()}
    counts = {}
    for split, folds in [("train", range(1, 9)), ("val", [9]), ("test", [10])]:
        rows = database[database.strat_fold.isin(folds)]  # in ascending ecg_id order
        signals = [
            wfdb.rdsamp(str(release / "records100" / "00000" / f"{ecg_id:05d}_lr"))[0].T
            for ecg_id in rows.index
        ]
        labels = [encoded[next(iter(ast.literal_eval(codes)))] for codes in rows.scp_codes]
        with h5py.File(raw / f"{split}_df.h5", "r") as f:
            np.testing.assert_array_equal(f["signals"][:], np.stack(signals))
            if split == "test":
                assert list(f) == ["signals"]
                stored = pd.read_csv(raw / "test_label_df.csv", index_col=0)["label"]
            else:
                stored = f["labels"][:]
        np.testing.assert_array_equal(stored, labels)
        counts[split] = len(rows)
    assert counts == {"train": 16, "val": 2, "test": 2}
