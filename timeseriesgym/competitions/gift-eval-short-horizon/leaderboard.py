"""Builds `leaderboard.csv` from GIFT-Eval's published results.

GIFT-Eval's leaderboard, the Hugging Face space Salesforce/GIFT-Eval, keeps one directory per
model under `results/`: `all_results.csv` holds the model's gluonts `MASE[0.5]` on every dataset
configuration, and `config.json` says whether the model saw test data (`testdata_leakage`).

Each model is scored as this challenge scores a submission: on each of the 5 datasets, its MASE
divided by Seasonal_Naive's, and the geometric mean of these ratios. The forecast windows are the
same, so the scores are comparable. Left out: models not marked `"testdata_leakage": "No"`, and
models without one finite, positive MASE on each of the 5 short-term configurations.

    python -m timeseriesgym.competitions.gift-eval-short-horizon.leaderboard

reads the space at REVISION (public and unauthenticated) and rewrites the leaderboard;
`--results-dir` builds it from a local copy of the space's `results/` instead.
"""

import argparse
import json
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .mase import normalized_mase

SPACE = "https://huggingface.co/spaces/Salesforce/GIFT-Eval"
SPACE_API = "https://huggingface.co/api/spaces/Salesforce/GIFT-Eval"
REVISION = "9d731e061162d5d8440204a2c455583d82e084d4"
REVISION_DATE = "2026-10-09 03:48:49"  # the revision's last modification, in UTC
# datasets.txt and the names of the same short-term configurations in GIFT-Eval's results
CONFIGURATIONS = {
    "LOOP_SEATTLE/5T": "loop_seattle/5T/short",
    "hierarchical_sales/W": "hierarchical_sales/W/short",
    "kdd_cup_2018_with_missing/H": "kdd_cup_2018/H/short",
    "m4_daily": "m4_daily/D/short",
    "solar/W": "solar/W/short",
}
METRIC = "eval_metrics/MASE[0.5]"
REFERENCE = "Seasonal_Naive"


def fetch_results(destination: Path, revision: str = REVISION) -> Path:
    """Downloads every `results/<model>/{all_results.csv,config.json}` of the space at
    `revision` into `destination`."""
    with urllib.request.urlopen(f"{SPACE_API}/revision/{revision}") as response:
        files = [entry["rfilename"] for entry in json.load(response)["siblings"]]
    for name in files:
        parts = name.split("/")
        if len(parts) != 3 or parts[0] != "results":
            continue
        if parts[2] not in ("all_results.csv", "config.json"):
            continue
        target = destination / parts[1] / parts[2]
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{SPACE}/resolve/{revision}/{urllib.parse.quote(name)}"
        with urllib.request.urlopen(url) as response:
            target.write_bytes(response.read())
    return destination


def read_results(directory: Path) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Each model directory's `all_results.csv` and `config.json`."""
    return {
        path.name: (
            pd.read_csv(path / "all_results.csv"),
            json.loads((path / "config.json").read_text()),
        )
        for path in sorted(directory.iterdir())
        if (path / "all_results.csv").is_file() and (path / "config.json").is_file()
    }


def _mase(rows: pd.DataFrame) -> dict[str, float] | None:
    """The model's MASE on each configuration, or None if one is missing, repeated or not a
    finite, positive number."""
    mase = {}
    for name, configuration in CONFIGURATIONS.items():
        values = pd.to_numeric(rows.loc[rows["dataset"] == configuration, METRIC], errors="coerce")
        if len(values) != 1 or not (np.isfinite(values.iloc[0]) and values.iloc[0] > 0):
            return None
        mase[name] = float(values.iloc[0])
    return mase


def build(results: dict[str, tuple[pd.DataFrame, dict]]) -> pd.DataFrame:
    """The leaderboard, best first, with the columns of the other leaderboards."""
    scored = {}
    for rows, config in results.values():
        if config.get("testdata_leakage") != "No":
            continue
        for model, model_rows in rows.groupby("model"):
            mase = _mase(model_rows)
            if mase is not None:
                scored[model] = mase
    reference = scored[REFERENCE]
    scores = {
        model: normalized_mase([mase[name] / reference[name] for name in CONFIGURATIONS])
        for model, mase in scored.items()
    }
    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    return pd.DataFrame(
        {
            "teamId": range(len(ranked)),
            "teamName": [model for model, _ in ranked],
            "submissionDate": REVISION_DATE,
            "score": [round(score, 5) for _, score in ranked],  # as the grader rounds
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, help="a local copy of the space's results/")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "leaderboard.csv")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as download:
        directory = args.results_dir or fetch_results(Path(download))
        build(read_results(directory)).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
