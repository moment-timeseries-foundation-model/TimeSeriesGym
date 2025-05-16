import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def prepare(raw: Path, public: Path, private: Path) -> None:
    del raw

    # Delete all test files
    os.remove(private / "test_labels.h5")
    os.remove(public / "test_features.h5")
    os.remove(public / "sample_submission.h5")

    # Load training features and labels
    with h5py.File(public / "train_features.h5", "r") as f:
        # List available leads
        print(f["train"].keys())

        # Load specific leads
        lead_1_train = f["train/lead_1"][()]

    with h5py.File(public / "train_labels.h5", "r") as f:
        labels = f["labels"][()]

    # Filter out heartbeats that are not normal or PVC
    normal_or_pvc = np.where(labels[:, 0] + labels[:, 1])[0]

    labels = labels[normal_or_pvc][:, :2]
    train_features = lead_1_train[normal_or_pvc]

    assert len(labels) == len(train_features), "Mismatch between number of labels and features"
    assert labels.shape == (
        len(train_features),
        2,
    ), "Labels should have two columns for normal and PVC"

    train_features = pd.DataFrame(train_features).reset_index(names="Id")
    train_features.to_csv(public / "train_features.csv", index=False)

    labels = pd.DataFrame(labels, columns=["normal", "pvc"]).reset_index(names="Id")
    labels.to_csv(private / "labels.csv", index=False)

    # Make all heartbeats normal for the sample submission
    sample_submission = labels.copy()
    sample_submission["normal"] = 1
    sample_submission["pvc"] = 0
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Remove the old HDF5 files
    os.remove(public / "train_features.h5")
    os.remove(public / "train_labels.h5")

    # Download Tutorials for Snorkel
    path_to_tutorial = "https://raw.githubusercontent.com/snorkel-team/snorkel-tutorials/refs/heads/master/spam/01_spam_tutorial.py"
    os.system(f"cd {public}; wget {path_to_tutorial}")

    # Download Snorkel documentation
    path_to_documentation = "https://readthedocs.org/projects/snorkel/downloads/pdf/latest/"
    os.system(f"cd {public}; curl -L {path_to_documentation} -o snorkel-docs.pdf")

    # Convert the PDF to text files, to make it easier for agents to read them
    os.system(f"cd {public}; pdftotext snorkel-docs.pdf snorkel-docs.txt")
    # NOTE: The page numbers are based on the PDF of Release 0.9.7
    os.system(f"cd {public}; pdftotext snorkel-docs.pdf -f 39 -l 67 snorkel-labeling-doc.txt")

    # Download the Snorkel GitHub repository
    os.system(f"cd {public}; git clone https://github.com/snorkel-team/snorkel.git")

    # Copy assets/ to public
    assets_path = "timeseriesgym/competitions/MIT-BIH-Arrhythmia-Weak-Supervision/assets"
    os.system(f"cp -r {assets_path} {public}")
