# Overview

## Description
GIFT-Eval is a comprehensive benchmark designed to evaluate general time series forecasting models across diverse datasets, promoting the advancement of zero-shot forecasting capabilities in foundation models. GIFT-Eval encompasses 24 datasets covering over 144,000 time series and 177 million data points, spanning seven domains, 10 frequencies, multivariate inputs, and prediction lengths ranging from short to long-term forecasts.

In this challenge, we will focus on the **short-term** and **pointwise** forecasts component of the GIFT-Eval benchmark.

## Evaluation

### Metric
Submissions are evaluated with GIFT-Eval's leaderboard metric for point forecasts: the Mean Absolute Scaled Error (MASE) on each dataset, divided by the MASE of the seasonal naive forecast on the same dataset, and aggregated over the datasets by the geometric mean. **Lower is better.** A perfect forecast scores 0 and the seasonal naive forecast scores 1.0, so a score below 1 beats the seasonal naive forecast.

The MASE of a forecast $F$ on dataset $d$ is computed over all of its forecast windows, and the score is the geometric mean of the ratios:

$$MASE_d(F) = \frac{1}{n_d} \sum_{(w,h)} \frac{\left| A_{w,h} - F_{w,h} \right|}{s_w} \qquad Score = \left( \prod_{d=1}^{D} \frac{MASE_d(F)}{MASE_d(SN)} \right)^{1/D}$$

Where:
- $D$ is the number of datasets listed in `datasets.txt`
- $A_{w,h}$ and $F_{w,h}$ are the actual and the forecasted value of window $w$ at step $h$
- $n_d$ is the number of observations of dataset $d$ (all of its forecast windows), excluding those whose actual value is missing
- $s_w$ is the scale of window $w$: the mean of $|y_t - y_{t-m}|$ over the window's history $y$ (its test input), over the pairs in which both values are present
- $m$ is the dataset's season, from its frequency as in GIFT-Eval (gluonts' `get_seasonality`): 288 for 5-minute data (one day), 24 for hourly data, and 1 for daily and weekly data
- $SN$ is the seasonal naive forecast: the last $m$ values of each window's history, repeated over the prediction length, after each missing value in the history is replaced by the last value present before it (at the start, by the first value present). Where $m = 1$, it is the naive forecast (the last value).

This is GIFT-Eval's evaluation (gluonts' `evaluate_model` with `MASE()`, `axis=None`, `mask_invalid_label=True`, `allow_nan_forecast=False` and the season above) and the normalization and geometric mean of its leaderboard. The forecast windows are GIFT-Eval's own, so the seasonal naive MASE of each dataset is the one GIFT-Eval publishes for it (for example 3.2784 on `m4_daily`). A window whose scale is 0 would not count, as in gluonts; no provided dataset has one. Predictions must be finite: NaN and infinite values make the submission invalid. The grader also logs the previous metric of this challenge, the MAPE averaged over the datasets, which does not affect the score.

### Leaderboard
The leaderboard lists the models published on GIFT-Eval's leaderboard (the Hugging Face space `Salesforce/GIFT-Eval`, revision `9d731e061162d5d8440204a2c455583d82e084d4` of 2026-10-09), each scored with this metric from its published MASE on the 5 datasets; the forecast windows are the same, so scores are comparable. Only models that GIFT-Eval marks as free of test-data leakage (`"testdata_leakage": "No"`) and that have a result on each of the 5 datasets are listed: 123 of its 143 models, including Seasonal_Naive (1.0) and Naive (1.02262); the other 20 are flagged for leakage.

# Dataset Description

## File and Data Field Descriptions
- **[Dataset]/**: There are individual subfolders that contain the original dataset files. We will provide loading scripts for each specific dataset.
- **datasets.txt**: Each line in the file `datasets.txt` specifies a dataset to run for the particular evaluation experiment.
- **sample_submission.h5**: This provides a sample submission format. Your submission must have exactly its field names (one per dataset in `datasets.txt`) and per-field shapes.

## Loading Data
It is **very important** that you load data using the `Dataset` class in the provided `dataset.py` script. To load data for all required datasets, do following

```python
with open('datasets.txt', 'r') as file:
    dataset_names = [line.strip() for line in file]

dataset = dict()

for dataset_name in dataset_names:
    data_path = '' #insert the data path that stores the dataset
    dataset[dataset_name] = Dataset(name=dataset_name, term='short', to_univariate=False, storage_path=data_path)
```

`Dataset` class provides the following properties:
- **training_dataset**: The training dataset.
- **validation_dataset**: The validation dataset.
- **test_data**: The test dataset.

Note that timeseries modality splits the data to train/val/test on the time dimension rather than the series dimension. Thus each of these splits utilize same series but different time indices. The training_dataset and validation_dataset are iterators where each element is a dictionary with the following keys:

- start: the start time of the window
- target: the target values of the window. If the timeseries is univariate, target is a 1D array of shape (length, ). Otherwise, it is a 2D array of shape (num_variables x length)
- item_id: the id of the series
- freq: The frequency of the series
- past_feat_dynamic_real: [If exists] the past feature dynamic real values of the window

To load data from train or validation dataset, you can do following. Here we use the example `m4_daily`

```python
train_iter = dataset['m4_daily'].training_dataset

for sample in train_iter:
    print(sample['target'].shape)
```

To load the test data that you are to use to make predictions, you can do following

```python
test_input_split_iter = dataset['m4_daily'].test_data.input

for sample in test_input_split_iter:
    print(sample['target'].shape)
```

### Important Information

Here are key details to help you successfully complete this task:

#### Data Structure
- All samples in the test input data contain the same keys as those in the training and validation sets for consistency.
- Again, It is **very important** that you load data using the `Dataset` class in the provided `dataset.py` script.

#### Prediction Requirements
- **Prediction Length**: The required future prediction length varies between datasets. You can determine the specific length for each dataset by accessing `dataset.prediction_length`.

#### Validation Set Structure
- Each validation sample consists of a series with length: (training series length + one prediction window length).
- If using learning-based local models, you may use this additional prediction window as labels or for hyperparameter tuning.
- **Important**: The prediction window begins immediately after the final timestamp of the validation window.

### Evaluation Method
- The benchmark implements a rolling evaluation policy.
- Note that input target shapes in the test set vary across samples.
- You may choose to either:
  - Truncate samples to achieve uniform length, or
  - Implement methods that can handle flexible input lengths.
  - Figure out your own creative solution!

### Plagiarism
- Please do not train on test input. We will detect such behavior which can disqualify your score.

## Submission format

You are required to format your submission in a numpy compound dataset and save as a `submission.h5` file.

```python
import h5py
submission_path = '' #insert your submission path

dt = []

for dataset_name in dataset_names:
    dataset = Dataset(
        dataset_name,
        term=Term.SHORT,
        to_univariate=False,
        storage_path=data_path,
    )
    prediction_length = dataset.prediction_length
    target_dim = dataset.target_dim
    num_samples = len(dataset.test_data.input)
    if target_dim == 1:  # univariate (all datasets in datasets.txt)
        shapes = (num_samples, prediction_length)
    else:
        shapes = (num_samples, target_dim, prediction_length)
    dt.append((dataset_name, np.float32, shapes))

dt = np.dtype(dt)

with h5py.File(f"{submission_path}/submission.h5", 'w') as f:
    data = np.zeros(1, dtype=dt)
    for dataset_name in dataset_names:
        #insert your prediction of shape (num_samples x num_variables x prediction_length) if multivariate
        #insert your prediction of shape (num_samples x prediction_length) if univariate
        dataset_prediction = None
        data[dataset_name][0] = np.array(dataset_prediction)

    f.create_dataset("submission", data=data)
```
