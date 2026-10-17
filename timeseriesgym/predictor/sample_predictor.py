"""Sample predictor: forecasts 0 for every value.

The grader imports this file (as predict.py) and calls predict(windows) with a list of windows;
each window is a dict with "dataset", "item_id", "start", "freq", "prediction_length" and
"target" (the context, a float32 array). See the description's Submission section.
"""

import numpy as np


def predict(windows):
    out = []
    for w in windows:
        shape = (w["prediction_length"], *w["target"].shape[1:])
        out.append(np.zeros(shape, dtype=np.float32))
    return out
