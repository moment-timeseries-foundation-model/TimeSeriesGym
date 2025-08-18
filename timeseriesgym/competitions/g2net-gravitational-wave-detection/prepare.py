import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def get_file_path(type: str, image_id):
    return f"{type}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.npy"


def prepare(raw: Path, public: Path, private: Path) -> None:
    random_seed = 0
    test_size = 0.1
    np.random.seed(random_seed)

    # remove test directory recursively from raw to save space
    shutil.rmtree(raw / "test")

    all_train_labels = pd.read_csv(raw / "training_labels.csv")

    # prepare new private test set
    private_test_labels = all_train_labels.sample(frac=test_size)
    private_test_ids = private_test_labels["id"].tolist()

    # prepare new public (train + test, depends how the user wants to split the data)
    public_labels = all_train_labels.drop(private_test_labels.index)
    public_ids = public_labels["id"].tolist()

    # save the public labels to the public directory
    public_labels.to_csv(public / "training_labels.csv", index=False)

    # save the private labels to the private directory
    private_test_labels.to_csv(private / "test_labels.csv", index=False)

    # move the train files to the public directory
    for id in public_ids:
        destination = public / get_file_path("train", id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(raw / get_file_path("train", id), destination)

    # move the private test files to public directory for agents to perform inference
    for id in private_test_ids:
        destination = public / get_file_path("test", id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(raw / get_file_path("train", id), destination)

    # generate sample submission
    sample_submission = pd.DataFrame(
        {"id": private_test_ids, "target": [0] * len(private_test_ids)}
    )
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
