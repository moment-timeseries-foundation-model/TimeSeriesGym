# Overview

## Description

In this competition, your task is to implement the STOMP algorithm for computing the Matrix Profile of univariate time series in Python, based on a provided implementation written in R.

**Important**: This challenge evaluates your ability to accurately translate a R-based implementation into a Python version, while adhering to the required specifications for the Python code. Since both your understanding of the original R code and your Python coding proficiency are critical to this task, please provide clear documentation for each step in your submission file.

## Code Requirements

Your Python implementation **must** define a function named `stomp`. Note that function names are case-sensitive—e.g., `Stomp` is not considered correct if the required function name is `stomp`. Below are the specifications:

### stomp

The `stomp` function computes the `Matrix Profile` and `Profile Index` for univariate time series using the STOMP algorithm.

- It supports both **self-join** (when only `ref_data` is provided) and **join similarity** (when both `ref_data` and `query_data` are provided).
- In the case of a self-join, the function also computes the **left** and **right** matrix profiles.

| Parameter           | Type                    | Description                                                                               |
|---------------------|-------------------------|-------------------------------------------------------------------------------------------|
| `ref_data`          | numpy.ndarray           | A univariate time series of shape (n_samples,).                                           |
| `query_data`        | numpy.ndarray or None   | Optional second univariate time series of shape (n_samples,). If provided, a join matrix profile is computed between `ref_data` and `query_data`. If None, a self-join is performed on `ref_data`.                                   |
| `window_size`       | int                     | Size of the sliding window.                                                               |
| `exclusion_zone`    | float, optional         | Size of the exclusion zone based on window size to avoid trivial matches during self-join. Default is 0.5. Ignored when `query_data` is provided.                                                                                      |
| `verbose`           | int, optional           | Controls verbosity of output: - `0`: No output - `1`: Text output - `2`: Text output with progress bar (default)                                                                                                                      |


| Returns   | Type                  | Description                                                                                           |
|-----------|-----------------------|-------------------------------------------------------------------------------------------------------|
| `mp`      | numpy.ndarray         | The matrix profile of shape (n_samples - window_size + 1,).                                           |
| `pi`      | numpy.ndarray         | The profile index corresponding to `mp`, same shape.  |
| `rmp`     | numpy.ndarray or None | Right matrix profile (used only in self-join). Same shape as `mp`. Returns None if `query_data` is provided (join similarity).                                                                                                                 |
| `rpi`     | numpy.ndarray or None | Right profile index. Same shape as `pi`. Returns None if `query_data` is provided (join similarity).  |
| `lmp`     | numpy.ndarray or None | Left matrix profile (used only in self-join). Same shape as `mp`. Returns None if `query_data` is provided (join similarity).                                                                                                                          |
| `lpi`     | numpy.ndarray or None | Left profile index. Same shape as `pi`. Returns None if `query_data` is provided (join similarity).   |


**Important**: You may define additional helper classes, methods, or functions as needed, but the required function above must strictly follow the specifications.


## Evaluation

Submissions will be evaluated based on the following three criteria:

- **Code Definition**: The required `stomp` function must be implemented and conform to the specifications outlined in **Code Requirements**.
- **Code Execution**: The `stomp` function must execute successfully and produce valid outputs when given inputs that meet the specifications.
- **Correctness**: The `stomp` function must return accurate outputs for both:
  - **self-join**: when only `ref_data` is provided.
  - **join similarity**: when both `ref_data` and `query_data` are provided.

The following sample code illustrates how outputs are obtained using your implemented `stomp` function:

```python
import numpy as np
from submission import stomp

# self-join
ref_data = np.load("ref_data.npy")
mp, pi, rmp, rpi, lmp, lpi = stomp(ref_data=ref_data, query_data=None, window_size=30, exclusion_zone=0.5, verbose=2)

# join similarity
query_data = np.load("query_data.npy")
mp, pi, rmp, rpi, lmp, lpi = stomp(ref_data=ref_data, query_data=query_data, window_size=30, exclusion_zone=0.5, verbose=2)
```

### Submission Format

You **MUST** submit your code in a single Python script named `submission.py` in your submission directory.


## File Descriptions

- **R/**: Contains the R implementation of the `stomp` algorithm. The main function is defined in `stomp.R`, but it may rely on helper functions defined in other files within the same directory, e.g., `dist_profile()` in `dist_profile.R`.
- **ref_data.npy**: A NumPy array (`numpy.ndarray`) containing a univariate time series to be used as `ref_data`.
- **query_data.npy**: A NumPy array (`numpy.ndarray`) containing a univariate time series to be used as `query_data`.
- **sample_submission.py**: A sample submission Python script containing the correct function definition, but with empty implementation.


## Acknowledgments

The R implementation of the STOMP algorithm is sourced from the [Time Series with Matrix Profile (tsmp) GitHub repository](https://github.com/matrix-profile-foundation/tsmp), which provides R functions implementing the [UCR Matrix Profile Algorithm](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html).
