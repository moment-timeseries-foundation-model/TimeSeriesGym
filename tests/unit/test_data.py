"""Downloading and preparing Kaggle competitions, offline: the Kaggle API is replaced by a fake that
serves a tiny synthetic zip of each competition's raw files. The real downloads are covered by
the integration tests."""

import dataclasses
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from timeseriesgym import data
from timeseriesgym.data import download_and_prepare_dataset, download_dataset
from timeseriesgym.registry import Registry
from timeseriesgym.utils import is_empty


def _amp_raw(raw: Path) -> None:
    visits = [(patient, month) for patient in range(10, 30) for month in (0, 6)]
    clinical = pd.DataFrame(
        {
            "visit_id": [f"{patient}_{month}" for patient, month in visits],
            "patient_id": [patient for patient, _ in visits],
            "visit_month": [month for _, month in visits],
            **{f"updrs_{part}": [part + month for _, month in visits] for part in range(1, 5)},
            "upd23b_clinical_state_on_medication": "On",
        }
    )
    clinical.to_csv(raw / "train_clinical_data.csv", index=False)
    clinical.head(0).to_csv(raw / "supplemental_clinical_data.csv", index=False)
    measured = clinical[["visit_id", "patient_id"]]
    measured.assign(Peptide="P").to_csv(raw / "train_peptides.csv", index=False)
    measured.assign(UniProt="U").to_csv(raw / "train_proteins.csv", index=False)
    (raw / "example_test_files").mkdir()
    pd.DataFrame(columns=["prediction_id", "rating", "group_key"]).to_csv(
        raw / "example_test_files" / "sample_submission.csv", index=False
    )
    pd.DataFrame(
        columns=["visit_id", "visit_month", "patient_id", "updrs_test", "row_id", "group_key"]
    ).to_csv(raw / "example_test_files" / "test.csv", index=False)


def _spaceship_titanic_raw(raw: Path) -> None:
    pd.DataFrame(
        {
            "PassengerId": [f"{i:04d}_01" for i in range(20)],
            "Age": range(20),
            "Transported": [i % 2 == 0 for i in range(20)],
        }
    ).to_csv(raw / "train.csv", index=False)


def _child_mind_institute_raw(raw: Path) -> None:
    series_ids = [f"series{i}" for i in range(10)]
    pd.DataFrame(
        {
            "series_id": np.repeat(series_ids, 100),
            "step": np.tile(np.arange(100), len(series_ids)),
            "anglez": 0.0,
            "enmo": 0.0,
        }
    ).to_parquet(raw / "train_series.parquet", index=False)
    pd.DataFrame(
        {
            "series_id": np.repeat(series_ids, 2),
            "night": 1,
            "event": ["onset", "wakeup"] * len(series_ids),
            "step": [20, 80] * len(series_ids),
        }
    ).to_csv(raw / "train_events.csv", index=False)


SYNTHETIC_RAW = {
    "amp-parkinsons-disease-progression-prediction": _amp_raw,
    "spaceship-titanic": _spaceship_titanic_raw,
    "child-mind-institute-detect-sleep-states": _child_mind_institute_raw,
}


class _FakeKaggleApi:
    """Stands in for an authenticated `KaggleApi`. A download writes the competition's synthetic
    raw files as `<competition>.zip`, the name Kaggle gives it, into the download directory."""

    def __init__(self, staging: Path):
        self.staging = staging
        self.downloads = []

    def competition_download_files(self, competition, path=None, force=False, quiet=True):
        self.downloads.append(competition)
        raw = self.staging / competition
        raw.mkdir(parents=True)
        SYNTHETIC_RAW[competition](raw)
        with ZipFile(Path(path) / f"{competition}.zip", "w") as zf:
            for fpath in sorted(raw.rglob("*")):
                if fpath.is_file():
                    zf.write(fpath, fpath.relative_to(raw))
        shutil.rmtree(raw)


@pytest.fixture
def kaggle_api(tmp_path, monkeypatch):
    # `download_dataset` imports `kaggle.rest`, and importing `kaggle` authenticates: with dummy
    # credentials the import works offline. They are never used, as the API is the fake.
    monkeypatch.setenv("KAGGLE_USERNAME", "offline")
    monkeypatch.setenv("KAGGLE_KEY", "offline")
    api = _FakeKaggleApi(tmp_path / "kaggle")
    monkeypatch.setattr(data, "authenticate_kaggle_api", lambda: api)
    return api


@pytest.mark.parametrize("competition_id", SYNTHETIC_RAW)
def test_downloading_competition_dataset_downloads_a_zip_file(
    competition_id: str,
    kaggle_api: _FakeKaggleApi,
    tmp_path: Path,
):
    registry = Registry(tmp_path / "data")

    download_dir = registry.get_data_dir() / competition_id
    zipfile = download_dataset(competition_id=competition_id, download_dir=download_dir)

    assert zipfile.is_file()
    assert zipfile.suffix == ".zip"
    assert zipfile.parent == download_dir
    assert kaggle_api.downloads == [competition_id]


@pytest.mark.parametrize("competition_id", SYNTHETIC_RAW)
def test_preparing_competition_dataset_creates_non_empty_public_and_private_directories(
    competition_id: str,
    kaggle_api: _FakeKaggleApi,
    tmp_path: Path,
):
    registry = Registry(tmp_path / "data")

    competition = registry.get_competition(competition_id)
    # the shipped checksums are those of the real data: write the synthetic data's elsewhere
    competition = dataclasses.replace(competition, checksums=tmp_path / "checksums.yaml")
    download_and_prepare_dataset(competition)

    assert competition.public_dir.is_dir(), "Public directory does not exist!"
    assert not is_empty(competition.public_dir), "Public directory is empty!"
    assert competition.private_dir.is_dir(), "Private directory does not exist!"
    assert not is_empty(competition.private_dir), "Private directory is empty!"
    assert kaggle_api.downloads == [competition_id]
