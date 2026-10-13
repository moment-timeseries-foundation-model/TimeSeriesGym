# Overview

## Description
GIFT-Eval is a comprehensive benchmark designed to evaluate general time series forecasting models across diverse datasets, promoting the advancement of zero-shot forecasting capabilities in foundation models. GIFT-Eval encompasses 24 datasets covering over 144,000 time series and 177 million data points, spanning seven domains, 10 frequencies, multivariate inputs, and prediction lengths ranging from short to long-term forecasts.

In this challenge, we will focus on the **short-term** and **pointwise** forecasts component of the GIFT-Eval benchmark.

## Evaluation

### Metric
Submissions are evaluated based on the Mean Absolute Percentage Error (MAPE) between forecasts and ground truth.

The MAPE is computed for each dataset and then averaged over the datasets:

$$MAPE = \frac{1}{D} \sum_{d=1}^{D} \frac{1}{n_d} \sum_{i=1}^{n_d} \left| \frac{A_{d,i} - F_{d,i}}{A_{d,i}} \right|$$

Where:
- $D$ is the number of datasets listed in `datasets.txt`
- $n_d$ is the number of observations of dataset $d$ (all of its forecast windows), excluding those whose actual value is missing
- $A_{d,i}$ is the actual value
- $F_{d,i}$ is the forecasted value

The score is a fraction, not a percentage, and each dataset counts equally regardless of its number of observations. The grader uses scikit-learn's `mean_absolute_percentage_error`, which divides by max(|A|, machine epsilon): a non-zero forecast where the actual value is exactly 0 makes that dataset's error very large. Predictions must not contain NaN values.

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
