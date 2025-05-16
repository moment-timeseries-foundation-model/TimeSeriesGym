import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

from timeseriesgym.utils import get_logger, read_csv

logger = get_logger(__name__)


def generate_random_id(length: int = 6):
    """Generate a random alphanumeric string of specified length."""
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return "".join(random.choice(characters) for _ in range(length))


# Notes:
# The training dataset consists of 16 seismic events (17 events)


def prepare(raw: Path, public: Path, private: Path) -> None:
    np.random.seed(0)  # Set random seed for reproducibility

    # Read the old training dataset
    logger.info("Reading the old training dataset...")
    old_train = read_csv(
        raw / "train.csv", dtype={"acoustic_data": np.int16, "time_to_failure": np.float64}
    )
    # Should take ~4 minutes

    # Find the start and end of each seismic event sequence
    t = old_train["time_to_failure"].to_numpy()
    t1 = np.roll(t, 1)
    temp = np.where(t[1:] - t1[1:] > 0)[0]

    n_segments = 17  # There are 16 events, 17 segments
    starts = np.zeros(n_segments)
    ends = np.ones(n_segments) * len(old_train)

    starts[1:] = temp + 1
    ends[:-1] = temp
    segments = list(zip(starts, ends, strict=False))

    # Train split includes the first 12 (~75%) segments, ~73% of the observations
    new_train = old_train[: int(segments[11][1])]

    # Create public files visible to agents
    logger.info("Saving the new training dataset...")
    new_train.to_csv(public / "train.csv", index=False)

    # Test split includes the last 5 (~25%) segments
    new_test_segments = []

    for start, end in segments[12:]:
        new_test_segments.append(old_train[int(start) : int(end)])

    # Sample 500 random contiguous sequences from each segment
    # Each sequence is 15000 observations long
    n_test_observations = 15000
    n_samples = 500  # Number of samples per segment
    seg_ids = []

    # First make a public test directory
    public_test = public / "test"
    public_test.mkdir(parents=True, exist_ok=True)

    logger.info("Saving the new testing datasets...")
    for segment in new_test_segments:
        for _ in range(n_samples):
            start = np.random.randint(0, len(segment) - n_test_observations)
            end = start + n_test_observations
            seg = segment[start:end]
            time_to_failure = seg.iloc[-1, 1]

            rand_id = generate_random_id()
            seg_ids.append((f"seg_{rand_id}", time_to_failure))

            # Save to public and private directories
            # seg.to_csv(private / f'seg_{rand_id}.csv', index=False)
            seg.drop(columns="time_to_failure").to_csv(
                public_test / f"seg_{rand_id}.csv", index=False
            )

    # Create sample submission CSV
    test_columns = ["seg_id", "time_to_failure"]
    pd.DataFrame(seg_ids, columns=test_columns).to_csv(private / "test.csv", index=False)

    sample_submission = pd.DataFrame([(i[0], 0) for i in seg_ids], columns=test_columns)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    logger.info("Saved sample submission CSV!")

    return
