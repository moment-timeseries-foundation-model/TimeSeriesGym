import zipfile
from copy import deepcopy
from pathlib import Path

from timeseriesgym.utils import get_logger, read_csv

logger = get_logger(__name__)


def prepare(raw: Path, public: Path, private: Path) -> None:
    stop_date = "2015-01-01"

    # first, for all file that ends with .zip, unzip it
    for file in raw.glob("*.zip"):
        logger.info(f"Unzipping {file}")
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(raw)
        # remove the zip file
        file.unlink()

    macro = read_csv(raw / "macro.csv")
    df = read_csv(raw / "train.csv")
    data_dictionary = raw / "data_dictionary.txt"

    train_df = df[df["timestamp"] < stop_date]
    test_df = df[df["timestamp"] >= stop_date]
    test_label_df = test_df[["id", "price_doc"]]
    test_df = test_df.drop(columns=["price_doc"])

    sample_submission = deepcopy(test_label_df)
    sample_submission["price_doc"] = 0

    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    train_df.to_csv(public / "train_features.csv", index=False)
    test_df.to_csv(public / "test_features.csv", index=False)
    test_label_df.to_csv(private / "test.csv", index=False)
    macro.to_csv(public / "macro.csv", index=False)

    # save the data dictionary to public
    with open(data_dictionary) as f:
        data_dict = f.read()
    with open(public / "data_dictionary.txt", "w") as f:
        f.write(data_dict)
