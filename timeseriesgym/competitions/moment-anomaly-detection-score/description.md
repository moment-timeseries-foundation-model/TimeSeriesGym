# MomentFM Anomaly Detection Challenge

## Description
In this challenge, you'll use the time-series foundation model [**`MOMENT`**](https://arxiv.org/pdf/2402.03885) to perform anomaly detection. `MOMENT` is a pre-trained encoder-only model designed for time-series representation learning.

Anomaly detection is commonly approached by reconstructing the input time series and computing an anomaly score based on the difference between the original and reconstructed signals. Your task is to use MOMENT to reconstruct the provided data, which will then be used to compute anomaly scores.

## Task

1. **Install the MomentFM package**
   ```bash
   pip install momentfm
   ```

2. **Explore the provided repository**
   The MOMENT repository has been cloned into your input data directory. You may find the following files helpful:

   * `tutorials/` – contains example notebooks to guide your implementation.
   * `README.md` – provides an overview of the `momentfm` package.
   * `momentfm/models/moment.py` – contains the model implementation.

3. **Load the data and compute reconstructions**
   Use the MOMENT model to reconstruct the provided input time series. These reconstructions will be used to calculate the anomaly scores. The anomaly score we will use is mean quared error (MSE) between original time series and the reconstruction.

## Files

* `data.npy`
  Contains 1000 univariate time series, each of length 512. Shape: `(1000, 512)`.

* `sample_submission.npy`
  A zero-filled placeholder array showing the required output shape.

## Submission

Submit a NumPy array named `anomaly_score.npy` containing the anomaly scores for each input series. The shape must match the input data: `(1000, 512)`.

## Evaluation

Your submission will be evaluated as follows:

* **Score = 100**: Output matches the reference exactly and is correctly saved.
* **Score = 0**: Output is missing, incorrectly formatted, or manually altered.
