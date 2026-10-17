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
- **[Dataset]/**: There are individual subfolders that contain the dataset files of the datasets in `datasets.txt`, at the frequencies listed there. Each series ends where its test windows begin: the test windows are not in your data directory, and the grader gives them to your predictor (see Submission). We will provide loading scripts for each specific dataset.
- **datasets.txt**: Each line in the file `datasets.txt` specifies a dataset to run for the particular evaluation experiment.
- **dataset.py**: the `Dataset` class (below).
- **sample_predictor/**: a valid predictor that forecasts 0 everywhere (see Submission).
- **check_predictor.py**, **predictor_tools/**: the local check of your predictor (see Submission).
- **predictor_datasets.json**: for each dataset, its frequency, prediction length, number of series, number of test windows per series and in all, and how many different times the test windows start at.

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
- **test_data**: The test dataset. In your data directory, each series ends where its hidden test windows begin (it is the series of `validation_dataset` of the full data), so here `test_data` gives the windows just before the hidden test windows: use them for validation. `Dataset.windows` counts them from the shorter series, so for some datasets there are fewer of them than test windows; `check_predictor.py` uses as many validation windows as there are test windows.

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

To iterate over the windows of `test_data` (here: the validation windows before the hidden test windows), you can do following

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
- The benchmark implements a rolling evaluation policy: the test windows of a series are its last `windows` spans of `prediction_length` values, and the input of each test window is every value of the series before it, including the actual values of the series' earlier test windows.
- Note that input target shapes in the test set vary across samples.
- You may choose to either:
  - Truncate samples to achieve uniform length, or
  - Implement methods that can handle flexible input lengths.
  - Figure out your own creative solution!

### Plagiarism
- Please do not train on test input. The test windows are not in your data directory: the grader gives them to your predictor.

## Submission

You do not submit predictions. You submit a **predictor**: the directory `/home/submission/predictor/`. After you finish, the grader builds the test windows with the `Dataset` class from the full series, `Dataset(name, term='short').test_data` of each dataset in `datasets.txt`, and runs your predictor on them.

**The directory**

- `predict.py` must define a function `predict(windows)` at module level. The grader imports `predict.py` once, as the module `predict`, with the predictor directory as the working directory and first on `sys.path` (so it can import its own modules), then calls `predict` many times in the same process.
- Put everything the predictor needs into the directory: code, model weights, statistics. It does not get your data directory; load files relative to `Path(__file__).parent`. At most 2 GiB and 10,000 files and directories in total; regular files and directories only (no symlinks).
- `sample_predictor/` in your data directory is a valid predictor that forecasts 0 everywhere.

**One call: `predict(windows)`**

`windows` is a list of dicts, one per test window, with:

- `dataset` (str): the dataset's name, as in `datasets.txt`.
- `item_id` (str): the series' `item_id`.
- `start` (str): the timestamp of `target[0]`, in ISO 8601 (`pd.Period(start, freq)` is GluonTS's `start` of the series).
- `freq` (str): the dataset's pandas frequency string.
- `prediction_length` (int): the number of time steps to forecast (the dataset's `prediction_length`).
- `target` (float32 NumPy array of shape (T,)): the input of the window in `test_data`: every value of the series before the forecast period. T varies; it may contain NaN.

`predict` returns one forecast per window, in the same order: a list of arrays of shape (prediction_length,) (or one array of shape (len(windows), prediction_length)), finite numbers only. Forecasts are stored as float32.

**The calls**

- The 16,696 test windows (table below) are sent in 547 rounds (568 calls), forward in time. Within a dataset, the windows go in order of the time their forecast period starts, and all windows that start at the same time are in the same round; the series of a dataset share one clock, and each dataset has its own. A call holds at most 1,024 windows and 33,554,432 input values and can mix datasets.
- So no call shows a value that lies, in time, in the forecast period of a window of the same call or of a later call, in any series of the same dataset: each test value reaches your predictor only after it has been forecast (later windows' inputs include the earlier windows' actual values).

| dataset | `freq` | `prediction_length` | series | test windows per series | test windows | forecast start times |
|---|---|---|---|---|---|---|
| `LOOP_SEATTLE/5T` | `5T` | 48 | 323 | 20 | 6,460 | 20 |
| `hierarchical_sales/W` | `W-WED` | 8 | 118 | 4 | 472 | 4 |
| `kdd_cup_2018_with_missing/H` | `H` | 48 | 270 | 20 | 5,400 | 40 |
| `m4_daily` | `D` | 14 | 4,227 | 1 | 4,227 | 547 |
| `solar/W` | `W-FRI` | 8 | 137 | 1 | 137 | 1 |

Each series in your data directory ends where its own test windows begin. In `kdd_cup_2018_with_missing/H`, 60 of the 270 series end 8 hours after the other 210, so they hold values from the first 8 hours of the other series' test period (other stations or pollutants, not their test values).

**Where it runs**

- The grader runs the predictor in a separate process, on CPU (do not rely on a GPU), with the thread counts of the usual numeric libraries (`OMP_NUM_THREADS` and others) set to 8. Use only the Python packages of this environment.
- The whole run, importing `predict.py` included, is stopped after 2,400 seconds (40 minutes).
- A predictor that fails to import, raises, returns the wrong number or shape of forecasts or a NaN or infinite value, or runs out of time is an invalid submission.

**Checking it**

- `python /home/data/check_predictor.py [/home/submission/predictor]` runs your predictor as the grader does, with the same format check, the same windows code, the same calls and the same metric, on validation windows built from your data directory (for each series, as many windows as it has test windows, ending where the test windows begin), and prints the score (the metric above) of each dataset and of all of them, and an estimate of the run time on the test windows. `--limit N` runs only the first N windows of each series, `--datasets m4_daily,solar/W` only some datasets.
- Only the final contents of `/home/submission/predictor/` are graded, after you finish.
