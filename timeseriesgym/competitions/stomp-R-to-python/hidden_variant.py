"""A variant of each stomp test case that the agent has no outputs for.

The variant's inputs are the public inputs reversed in time and transformed by a * x + b. The
z-normalized distance between two windows does not change when both are reversed or both are
transformed by the same a * x + b (a != 0), so the R outputs of the public case give the
expected outputs of the variant exactly: the profiles are reversed, a 1-based index i becomes
n + 1 - i (n windows; -Inf, "never updated", stays -Inf), and the left and right profiles swap.
Outputs computed offline for the public inputs and pasted into the code fail the variant.
"""

import numpy as np

SCALE, SHIFT = -1.5, 0.5


def transform(series: np.ndarray, scale: float = SCALE, shift: float = SHIFT) -> np.ndarray:
    """The variant's input for a public input series."""
    return scale * np.asarray(series, dtype=float)[::-1] + shift


def _reverse_index(index: np.ndarray, n_windows: int) -> np.ndarray:
    reversed_ = np.asarray(index, dtype=float)[::-1].copy()
    finite = np.isfinite(reversed_)
    reversed_[finite] = n_windows + 1 - reversed_[finite]
    return reversed_


def expected_self_join(outputs: np.ndarray) -> np.ndarray:
    """Expected (mp, pi, rmp, rpi, lmp, lpi) columns of the variant, from the public case's."""
    mp, pi, rmp, rpi, lmp, lpi = np.asarray(outputs, dtype=float).T
    n = len(mp)
    return np.column_stack(
        [
            mp[::-1],
            _reverse_index(pi, n),
            lmp[::-1],  # the right profile of the reversed series is the reversed left one
            _reverse_index(lpi, n),
            rmp[::-1],
            _reverse_index(rpi, n),
        ]
    )


def expected_join(outputs: np.ndarray, n_query_windows: int) -> np.ndarray:
    """Expected (mp, pi) columns of the variant, from the public case's; pi indexes the query's
    `n_query_windows` windows."""
    mp, pi = np.asarray(outputs, dtype=float).T
    return np.column_stack([mp[::-1], _reverse_index(pi, n_query_windows)])
