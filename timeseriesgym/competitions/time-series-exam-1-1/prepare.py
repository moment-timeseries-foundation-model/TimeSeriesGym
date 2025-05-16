from pathlib import Path

from timeseriesgym.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # read and send sample submission to public
    example_submission = read_csv(raw / "sample_submission.csv")
    example_submission.to_csv(public / "sample_submission.csv", index=False)

    # read and send answer to private
    answer = read_csv(raw / "test.csv")
    answer.to_csv(private / "test.csv", index=False)

    # read and send train to public
    train = read_csv(raw / "qa_dataset.csv")
    train.to_csv(public / "qa_dataset.csv", index=False)
