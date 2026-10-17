import hashlib
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

from timeseriesgym.utils import get_logger, read_csv

logger = get_logger(__name__)

# Kaggle's train.csv is one continuous run of 17 earthquake cycles with 16 laboratory earthquakes:
# the run starts and ends inside a cycle. time_to_failure falls within a cycle and jumps up after
# each earthquake.
N_CYCLES = 17
# The new train.csv holds the first 12 cycles (73% of the rows). The test segments are cut from
# the last 5 cycles.
N_TRAIN_CYCLES = 12
# Kaggle's test segments had 150,000 consecutive rows of acoustic_data each.
SEGMENT_ROWS = 150_000
# The signal was recorded in bins of 4,096 rows, and every cycle ends on a bin boundary. Segments
# cut back to back from a cycle's end would carry their distance to the earthquake in where the
# bin boundaries fall inside them, and neighbours would continue each other. So no two segments,
# and no segment and the start of its cycle, are less than one bin apart (segment_bounds).
BIN_ROWS = 4_096
# train.csv has 629,145,480 rows, so it is read in chunks.
CHUNK_ROWS = 10_000_000
DTYPES = {"acoustic_data": np.int16, "time_to_failure": np.float64}


def generate_random_id(rng: random.Random, length: int = 6) -> str:
    """Generate a random alphanumeric string of specified length."""
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return "".join(rng.choice(characters) for _ in range(length))


def find_earthquakes(path: Path) -> tuple[list[int], int]:
    """The last row of every cycle that ends in an earthquake, and the number of rows."""
    earthquakes = []
    offset, last = 0, None
    for chunk in read_csv(
        path,
        usecols=["time_to_failure"],
        dtype={"time_to_failure": np.float64},
        chunksize=CHUNK_ROWS,
    ):
        t = chunk["time_to_failure"].to_numpy()
        if last is not None and t[0] > last:  # the jump falls between two chunks
            earthquakes.append(offset - 1)
        earthquakes.extend((offset + np.flatnonzero(np.diff(t) > 0)).tolist())
        offset, last = offset + len(t), t[-1]
    return earthquakes, offset


def segment_bounds(
    cycles: list[tuple[int, int]], segment_rows: int, rng: random.Random
) -> list[tuple[int, int]]:
    """Non-overlapping `(start, end)` row ranges of `segment_rows` rows inside each cycle.

    Each cycle is cut backwards. Its last segment ends a random 0 to BIN_ROWS - 1 rows before the
    cycle's last row, and each earlier segment ends a random BIN_ROWS to 2 * BIN_ROWS - 1 rows
    before the next one starts. A segment that would start less than BIN_ROWS rows after the
    cycle's first row is dropped. No segment spans an earthquake, and the bin phase of a segment
    says nothing about its distance to the earthquake.
    """
    bounds = []
    for start, end in cycles:
        seg_end = end - rng.randint(0, BIN_ROWS - 1)
        while seg_end - segment_rows >= start + BIN_ROWS:
            bounds.append((seg_end - segment_rows, seg_end))
            seg_end -= segment_rows + rng.randint(BIN_ROWS, 2 * BIN_ROWS - 1)
    return bounds


def private_seed(values: np.ndarray) -> int:
    """A seed from a digest of private values, so that the public code alone cannot reproduce
    the draws."""
    return int.from_bytes(hashlib.sha256(values.astype("<f8").tobytes()).digest()[:8], "big")


def prepare(raw: Path, public: Path, private: Path) -> None:
    logger.info("Finding the earthquakes in the old training dataset...")
    earthquakes, n_rows = find_earthquakes(raw / "train.csv")
    if len(earthquakes) != N_CYCLES - 1:
        raise ValueError(
            f"expected {N_CYCLES - 1} earthquakes in train.csv, found {len(earthquakes)}"
        )

    # train.csv: cycles 1-12 as one continuous signal. It stops one row before the 12th
    # earthquake, as it always has, so the file is unchanged.
    train_end = earthquakes[N_TRAIN_CYCLES - 1]
    # The test: cycles 13-17, from the row after the 12th earthquake. No test row is in train.csv.
    test_start = earthquakes[N_TRAIN_CYCLES - 1] + 1

    logger.info("Saving the new training dataset and reading the test cycles...")
    test_x, test_t = [], []
    offset = 0
    with open(public / "train.csv", "w", newline="") as train_file:
        for chunk in read_csv(raw / "train.csv", dtype=DTYPES, chunksize=CHUNK_ROWS):
            n_train = min(len(chunk), max(0, train_end - offset))
            if n_train:
                chunk.iloc[:n_train].to_csv(train_file, header=offset == 0, index=False)
            first = max(0, test_start - offset)
            if first < len(chunk):
                test_x.append(chunk["acoustic_data"].to_numpy()[first:])
                test_t.append(chunk["time_to_failure"].to_numpy()[first:])
            offset += len(chunk)
    x, t = np.concatenate(test_x), np.concatenate(test_t)

    # Test segments: 150,000-row segments inside the test cycles with random gaps between them,
    # labelled with time_to_failure at their last row. The last cycle ends with the run, not an
    # earthquake. The gaps are drawn from a generator seeded with a digest of the test cycles'
    # time_to_failure, which is private, so the public code cannot reproduce them.
    starts = [e + 1 - test_start for e in earthquakes[N_TRAIN_CYCLES - 1 :]]
    ends = [e + 1 - test_start for e in earthquakes[N_TRAIN_CYCLES:]] + [n_rows - test_start]
    gap_rng = random.Random(private_seed(t))
    bounds = segment_bounds(list(zip(starts, ends, strict=True)), SEGMENT_ROWS, gap_rng)
    labels = t[[end - 1 for _, end in bounds]]

    # The ids must not tell where a segment lies in the run, even to someone who has this code:
    # their generator is seeded with a digest of the labels, which are private. The segments are
    # written, and listed in the answers and the sample submission, in id order.
    id_rng = random.Random(private_seed(labels))
    seg_ids, used_ids = [], set()
    for _ in bounds:
        rand_id = generate_random_id(id_rng)
        while rand_id in used_ids:
            rand_id = generate_random_id(id_rng)
        used_ids.add(rand_id)
        seg_ids.append(f"seg_{rand_id}")
    order = sorted(range(len(bounds)), key=seg_ids.__getitem__)

    logger.info(f"Saving {len(bounds)} test segments of {SEGMENT_ROWS} rows...")
    public_test = public / "test"
    public_test.mkdir(parents=True, exist_ok=True)
    for i in order:
        start, end = bounds[i]
        pd.DataFrame({"acoustic_data": x[start:end]}).to_csv(
            public_test / f"{seg_ids[i]}.csv", index=False
        )

    answers = pd.DataFrame(
        {"seg_id": [seg_ids[i] for i in order], "time_to_failure": labels[order]}
    )
    answers.to_csv(private / "test.csv", index=False)

    sample_submission = answers.assign(time_to_failure=0)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    logger.info("Saved sample submission CSV!")
