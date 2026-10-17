import logging
import shutil
from pathlib import Path

import gdown
import h5py
import numpy as np

from timeseriesgym.predictor import prepare as predictor_prepare
from timeseriesgym.predictor import windows as pw
from timeseriesgym.predictor.grading import check_windows
from timeseriesgym.utils import clone_repository, extract, load_answers


def download(raw: Path) -> None:
    """The datasets (raw/<group>/<name>.csv) and the TSL repository (raw/Time-Series-Library,
    without its .json and .csv files)."""
    # Download the long-term forecast dataset from Google Drive by TSL authors
    url = "https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322"
    # gdown returns None when it cannot list the folder; failures must stop the prepare
    if not gdown.download_folder(url=url, output=str(raw), quiet=False, use_cookies=False):
        raise RuntimeError(f"Failed to download the dataset from Google Drive: {url}")

    # Unzip the downloaded files in the long_term_forecast directory
    zip_dir = raw
    for zip_path in zip_dir.glob("*.zip"):
        extract_dir = zip_dir  # Create folder with same name as zip (without .zip)

        logging.info(f"Unzipping {zip_path.name} to {extract_dir}...")
        extract(zip_path, extract_dir)

    # Download the tsl GitHub repository (again, if an earlier prepare left a copy in raw)
    shutil.rmtree(raw / "Time-Series-Library", ignore_errors=True)
    clone_repository(
        "https://github.com/raycai420/Time-Series-Library.git",
        raw / "Time-Series-Library",
        branch="long-horizon-forecast",
        commit="7f9820b0cc67c6ae12307df2cc0a348a7721b5ff",
    )

    # Remove all .json and .csv files from the cloned repository
    for file in (raw / "Time-Series-Library").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()


def get_true_array(ds_name, pred_len=96, seq_len=96, root_path=None, data_path=None):
    """The labels of TSL's test split: `batch_y` of its test loader."""
    from data_provider.data_factory import data_provider

    class Arg:
        def __init__(self):
            self.data = "custom" if "ett" not in ds_name.lower() else ds_name
            self.pred_len = pred_len
            self.seq_len = seq_len
            self.root_path = root_path
            self.features = "M"
            self.embed = "-"
            self.batch_size = 512
            self.freq = "-"
            self.task_name = "long_term_forecast"
            self.data_path = data_path
            self.label_len = 0
            self.target = "OT"
            self.seasonal_patterns = None
            self.num_workers = 1

    args = Arg()
    data_set, data_loader = data_provider(args, "test")

    trues = []
    for _, (_, batch_y, _, _) in enumerate(data_loader):
        batch_y = batch_y[:, -args.pred_len :, :]
        trues.append(batch_y)

    trues = np.concatenate(trues, axis=0)
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    return trues


def prepare(raw: Path, public: Path, private: Path):
    """The agent gets each CSV up to TSL's first test label and submits a predictor; the grader
    keeps the full CSVs and TSL in `private` and builds the test windows from them (see
    timeseriesgym/predictor). The answers are unchanged."""
    download(raw)

    # the full data, from which the grader builds the test windows
    groups = sorted({group for group, _ in pw.TSL_DATASETS})
    for group in groups:
        shutil.copytree(raw / group, private / group)
    shutil.copytree(raw / "Time-Series-Library", private / "Time-Series-Library")
    found = sorted((p.parent.name, p.stem) for g in groups for p in (private / g).glob("*.csv"))
    if found != sorted(pw.TSL_DATASETS):
        raise ValueError(f"unexpected datasets: {found}")

    # 4. Run the data loading logic
    ds = {}
    with pw.tsl_imported(private / "Time-Series-Library"):
        for group, ds_name in pw.TSL_DATASETS:
            trues = get_true_array(
                ds_name,
                pred_len=96,
                seq_len=96,
                root_path=private / group,
                data_path=f"{ds_name}.csv",
            )

            trues = trues.astype(np.float16)
            ds[ds_name] = trues
            logging.info(f"Processed {ds_name} with shape {trues.shape}")

    # Directory containing your .npy files
    output_h5_file = private / "test.h5"
    with h5py.File(output_h5_file, "w") as h5f:
        submission_group = h5f.create_group("labels")  # Top-level dummy layer
        for ds_name, data in sorted(ds.items()):  # sorted: the same file on every host
            # Create a dataset for each .npy file
            submission_group.create_dataset(ds_name, data=data)
    del ds

    # The test windows, built from the full CSVs as the grader builds them, are the answers.
    repo = private / "Time-Series-Library"
    test = pw.tsl_windows(private, repo, "test", with_labels=True)
    check_windows(test, load_answers(output_h5_file))

    # public: each CSV up to the row of the first test label, and a copy of TSL whose loader
    # keeps the full files' borders and scaler
    meta, full_rows = [], {}
    for (group, name), d in zip(pw.TSL_DATASETS, test.datasets, strict=True):
        n_rows = d.extra["n_rows"]
        keep = pw.tsl_test_start(name, n_rows)
        assert keep == d.extra["border1"] + pw.SEQ_LEN, name
        predictor_prepare.cut_csv(
            private / group / f"{name}.csv", public / group / f"{name}.csv", keep
        )
        full_rows[f"{name}.csv"] = n_rows
        meta.append(
            {
                "name": name,
                "csv": f"{group}/{name}.csv",
                "freq": d.freq,
                "n_vars": d.out_shape[1],
                "rows": keep,
                "full_rows": n_rows,
                "context_length": pw.SEQ_LEN,
                "prediction_length": pw.PRED_LEN,
                "test_windows": d.n_windows,
            }
        )
    shutil.copytree(repo, public / "Time-Series-Library")
    loader = public / "Time-Series-Library" / "data_provider" / "data_loader.py"
    predictor_prepare.patch_tsl_loader(loader, full_rows)

    # The public CSVs end where the test labels begin; the local check's validation windows
    # are those of the full data.
    rows = {m["name"]: m["full_rows"] for m in meta}
    full_val = pw.tsl_windows(private, repo, "val", with_labels=True)
    public_val = pw.tsl_windows(
        public, public / "Time-Series-Library", "val", rows, with_labels=True
    )
    for m, a, b in zip(meta, full_val.datasets, public_val.datasets, strict=True):
        assert a.n_windows == b.n_windows and np.array_equal(a.labels, b.labels), m["name"]
        m["validation_windows"] = b.n_windows
    a_x = [w.target for w in full_val.windows]
    assert all(np.array_equal(x, w.target) for x, w in zip(a_x, public_val.windows, strict=True))
    del full_val, public_val, a_x

    config = predictor_prepare.predictor_config(Path(__file__).parent)
    predictor_prepare.write_common(
        public, Path(__file__).parent / "grade.py", config, {"datasets": meta}, test.windows
    )
    logging.info(f"Wrote the public data (CSVs cut at the test split) to {public}")
