import logging
import shutil
from pathlib import Path

import h5py
import numpy as np

from timeseriesgym.predictor import prepare as predictor_prepare
from timeseriesgym.predictor import windows as pw
from timeseriesgym.predictor.grading import check_windows
from timeseriesgym.utils import load_answers

from .mase import window_baselines


def _baselines(raw: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    from .dataset import Dataset  # needs gluonts and datasets: imported only to prepare

    return window_baselines(Dataset(name=name, term="short", to_univariate=False, storage_path=raw))


def prepare(raw: Path, public: Path, private: Path):
    """The agent gets every series up to its first test window and submits a predictor; the
    grader keeps the full series in `private` and builds the test windows from them (see
    timeseriesgym/predictor). The answers are unchanged."""
    datasets_path = Path(__file__).parent / "datasets.txt"
    dataset_prepare_path = Path(__file__).parent / "dataset.py"

    # the full data, from which the grader builds the test windows
    shutil.copy2(datasets_path, private / datasets_path.name)
    shutil.copy2(dataset_prepare_path, private / dataset_prepare_path.name)
    logging.info(f"Copying {datasets_path.name} and {dataset_prepare_path.name} to {private}")

    with open(private / datasets_path.name) as f:
        dataset_names = [line.strip() for line in f.readlines()]

    logging.info(f"Dataset names: {dataset_names}")

    # make new test.h5 from the raw data
    with (
        h5py.File(raw / "test.h5", "r") as original_file,
        h5py.File(private / "test.h5", "w") as subset_file,
    ):
        # Load the original data
        original_data = original_file["label"][()]

        # Get all dataset names from the original file
        all_dataset_names = original_data.dtype.names

        # Filter for only the datasets in our subset
        valid_subset_names = [name for name in dataset_names if name in all_dataset_names]

        # Create a new compound dtype with only the subset fields
        subset_dt = []
        subset_data = {}

        # Collect data and prepare dtype
        for name in valid_subset_names:
            # Get shape from original data
            shape = original_data[name][0].shape
            subset_dt.append((name, np.float32, shape))
            subset_data[name] = original_data[name][0]

        # Create the compound dtype
        subset_dt = np.dtype(subset_dt)

        # Create and fill subset data array
        data = np.zeros(1, dtype=subset_dt)
        for name in valid_subset_names:
            data[name][0] = subset_data[name]

        # GIFT-Eval's MASE needs each window's history: keep its scale and seasonal naive
        # forecast with the answers (the history itself is public)
        scale_dt, scale_data, naive_data = [], {}, {}
        for name in valid_subset_names:
            scale_data[name], naive_data[name] = _baselines(raw, name)
            if naive_data[name].shape != subset_data[name].shape:
                raise ValueError(f"The test windows of {name} do not match its answers.")
            scale_dt.append((name, np.float64, scale_data[name].shape))
        scales = np.zeros(1, dtype=np.dtype(scale_dt))
        naive = np.zeros(1, dtype=subset_dt)
        for name in valid_subset_names:
            scales[name][0] = scale_data[name]
            naive[name][0] = naive_data[name]

        # Save to the new file
        labels = subset_file.create_group("labels")
        labels.create_dataset("target", data=data)
        labels.create_dataset("seasonal_error", data=scales)
        labels.create_dataset("seasonal_naive", data=naive)
        logging.info(f"Created {private / 'test.h5'} with subset data.")

    if valid_subset_names != dataset_names:
        raise ValueError(f"datasets.txt lists datasets without answers: {dataset_names}")

    # Only the frequencies in datasets.txt: the others of the same datasets cover the test
    # period (the solar/W test labels are the weekly sums of solar/D, solar/H and solar/10T).
    for ds_name in dataset_names:
        shutil.copytree(raw / ds_name, private / ds_name)

    # The test windows, built from the full series as the grader builds them, are the answers.
    test = pw.gift_windows(private, private / "dataset.py", dataset_names, with_labels=True)
    check_windows(test, load_answers(private / "test.h5"))

    # public: each series up to its first test window (the series of `validation_dataset` of
    # the full data), in a form the agent environment's `datasets` reads
    module = pw.load_gift_dataset_module(private / "dataset.py")
    meta = []
    for ds_name in dataset_names:
        ds = module.Dataset(name=ds_name, term="short", storage_path=str(private))
        h, w = int(ds.prediction_length), int(ds.windows)
        lengths = [len(entry["target"]) for entry in ds.hf_dataset]
        cut = [n - h * w for n in lengths]
        if min(cut) < 1:
            raise ValueError(f"{ds_name}: a series is shorter than its test windows")
        predictor_prepare.cut_gift_dataset(private / ds_name, public / ds_name, cut)
        info = test.by_name[ds_name]
        meta.append(
            {
                "name": ds_name,
                "freq": str(ds.freq),
                "prediction_length": h,
                "series": len(lengths),
                "test_windows_per_series": w,
                "test_windows": info.n_windows,
                "forecast_starts": len({x.clock for x in test.windows if x.dataset == ds_name}),
                "public_length": {"min": min(cut), "max": max(cut)},
            }
        )
    shutil.copy2(datasets_path, public / datasets_path.name)
    predictor_prepare.patch_gift_dataset_py(dataset_prepare_path, public / "dataset.py")
    _check_public(private, public, test, {m["name"]: m["test_windows_per_series"] for m in meta})
    for m in meta:
        m["validation_windows"] = m["test_windows"]
    config = predictor_prepare.predictor_config(Path(__file__).parent)
    predictor_prepare.write_common(
        public, Path(__file__).parent / "grade.py", config, {"datasets": meta}, test.windows
    )
    logging.info(f"Wrote the public data (series cut before the test windows) to {public}")


def _check_public(private: Path, public: Path, test: pw.WindowSet, windows: dict) -> None:
    """Each public series is the full series up to its first test window, and the local check's
    validation windows (the test's count, one test span earlier) end right there."""
    numpy_had_object = "object" in vars(np)  # the public dataset.py adds it for datasets 2.1.0
    full = pw.load_gift_dataset_module(private / "dataset.py").Dataset
    cut = pw.load_gift_dataset_module(public / "dataset.py").Dataset
    names = [d.name for d in test.datasets]
    firsts = {}
    for w in test.windows:
        firsts.setdefault((w.dataset, w.series), w.t0)
    for name in names:
        a_ds = full(name=name, term="short", storage_path=str(private)).hf_dataset
        b_ds = cut(name=name, term="short", storage_path=str(public)).hf_dataset
        for s, (a, b) in enumerate(zip(a_ds, b_ds, strict=True)):
            t, p = np.asarray(a["target"]), np.asarray(b["target"])
            assert len(p) == firsts[(name, s)], (name, s)
            assert np.array_equal(t[: len(p)], p, equal_nan=True), (name, s)
            assert a["item_id"] == b["item_id"] and a["start"] == b["start"], (name, s)
    val = pw.gift_windows(public, public / "dataset.py", names, windows=windows)
    for d, v in zip(test.datasets, val.datasets, strict=True):
        assert v.n_windows == d.n_windows, (d.name, v.n_windows, d.n_windows)
    if not numpy_had_object:
        del np.object
