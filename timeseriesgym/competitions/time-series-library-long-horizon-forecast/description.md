# Overview

## Description

In this challenge, you will be evaluated on your ability to leverage a research repository to solve a time series forecasting task. We provide the popular research repository **Time Series Library (TSL)**, which implements several state-of-the-art forecasting models.

Your objective is to tackle the [Long Term Forecasting Challenge (LTF)](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon2.html#weather), using the models implemented in TSL. You are also welcome to build and use your own models. The goal is to produce highly accurate forecasts on the provided datasets.

### Your tasks:

1. Configure your model of choice (either from TSL or custom-built).
2. Train and evaluate it on each of the datasets in the Long Term Forecasting Challenge.

## Task

1. **Install the Time Series Library (TSL)**

   * The TSL repository is located in your input data directory.
   * To install dependencies:

   ```bash
   cd Time-Series-Library
   pip install -r requirements.txt
   ```

2. **Run the training/evaluation**

   * You could use `run.py` to train and evaluate your model with the following settings:

     * Lookback window: 96
     * Forecast horizon: 96
     * Label len: 0 **NOTE** this refers to the `label_len` hyperparameter used by TSL. If you use a different setup, your grade will be invalid.

   * The raw datasets included in the LTF challenge include multiple dataset. Each one is provided in the `datagroup_name/dataset_name.csv` under your data directory. You will need to configure the data loader accordingly.
   * Each CSV here ends where TSL's test split begins: the test rows are withheld, and the grader runs your predictor on them (see Submission). TSL's `data_provider/data_loader.py` is patched so that the training and validation splits and the scaler are those of the full files (`TSG_FULL_ROWS` records their row counts), and `flag="test"` gives the validation split. So `run.py` reports validation results where it would report test results.
   * Set an output path to save the experiment results.
   * **Hint:** Check `run.py` for available command-line arguments. Example scripts in the `scripts/` folder may also be helpful.

## Submission

You do not submit predictions. You submit a **predictor**: the directory `/home/submission/predictor/`. After you finish, the grader builds TSL's test windows from the full CSVs (lookback 96, horizon 96, `label_len` 0, `--features M`, target `OT`; they are not in your data directory) and runs your predictor on them.

**The directory**

* `predict.py` must define a function `predict(windows)` at module level. The grader imports `predict.py` once, as the module `predict`, with the predictor directory as the working directory and first on `sys.path` (so it can import its own modules, e.g. a copy of TSL's `models/` and `layers/`), then calls `predict` many times in the same process.
* Put everything the predictor needs into the directory: code, model weights (checkpoints), statistics. It does not get your data directory; load files relative to `Path(__file__).parent`. At most 2 GiB and 10,000 files and directories in total; regular files and directories only (no symlinks).
* `sample_predictor/` in your data directory is a valid predictor that forecasts 0 everywhere.

**One call: `predict(windows)`**

`windows` is a list of dicts, one per window, with:

* `dataset` (str): the dataset name, the CSV file stem (e.g. `ETTh1`).
* `item_id` (str): the dataset name again (each dataset is one multivariate series).
* `start` (str): the timestamp of `target[0]` (the CSV's `date`), in ISO 8601.
* `freq` (str): the pandas frequency of the CSV's dates (e.g. `h`, `15min`).
* `prediction_length` (int): 96.
* `target` (float32 NumPy array of shape (96, n_vars)): TSL's lookback window `seq_x`, the 96 time steps before the forecast period, standardized with the scaler that TSL fits on the training split (as `data_provider` returns it), the variables in TSL's column order (the CSV's columns without `date`, with `OT` moved last).

`predict` returns one forecast per window, in the same order: a list of arrays of shape (96, n_vars) (or one array of shape (len(windows), 96, n_vars)), finite numbers only, on the same standardized scale (do not inverse-transform). Forecasts are stored as float32.

**The calls**

* The test windows are TSL's test split of each dataset, with stride 1: 48,962 windows in all (table below). They are sent in 11,425 calls, forward in time: call k holds, for each dataset, the window whose forecast period starts at the k-th step of its clock. The four ETT datasets share one 15-minute clock (ETTh1 and ETTh2 are ETTm1 and ETTm2 sampled at the hour), so an ETTh window comes with every fourth ETTm window; every other dataset has its own clock. A call holds at most one window per dataset.
* So no call shows a value that lies, in time, in the forecast period of a window of the same call or of a later call, in any dataset on the same clock: each test value reaches your predictor only after it has been forecast.
* Do not keep information between calls: as in the task's setup (lookback 96), each forecast must depend only on its own window and on what you trained beforehand.

| dataset | `freq` | n_vars | test windows | validation windows |
|---|---|---|---|---|
| ETTh1 | `h` | 7 | 2,785 | 2,785 |
| ETTh2 | `h` | 7 | 2,785 | 2,785 |
| ETTm1 | `15min` | 7 | 11,425 | 11,425 |
| ETTm2 | `15min` | 7 | 11,425 | 11,425 |
| electricity | `h` | 321 | 5,165 | 2,537 |
| exchange_rate | `D` | 8 | 1,422 | 665 |
| national_illness | `W-TUE` | 7 | 98 | 2 |
| traffic | `h` | 862 | 3,413 | 1,661 |
| weather | `10min` | 21 | 10,444 | 5,175 |

**Where it runs**

* The grader runs the predictor in a separate process, on CPU (do not rely on a GPU), with the thread counts of the usual numeric libraries (`OMP_NUM_THREADS` and others) set to 8. Use only the Python packages of this environment.
* The whole run, importing `predict.py` included, is stopped after 2,400 seconds (40 minutes).
* A predictor that fails to import, raises, returns the wrong number or shape of forecasts or a NaN or infinite value, or runs out of time is an invalid submission.

**Checking it**

* `python /home/data/check_predictor.py [/home/submission/predictor]` runs your predictor as the grader does, with the same format check, the same windows code, the same calls and the same metric, on the windows of TSL's validation split (`flag="val"`) of your data, and prints the MSE of each dataset and an estimate of the run time on the test windows. `--limit N` runs only the first N windows of each dataset, `--datasets ETTh1,weather` only some datasets.
* `predictor_datasets.json` in your data directory lists, for each dataset, its CSV, frequency, number of variables, row counts (here and in the full file) and numbers of test and validation windows.
* Only the final contents of `/home/submission/predictor/` are graded, after you finish.

## Data

Your `data` directory includes the following:

* **Time-Series-Library/** – the cloned TSL repository, with the patched `data_provider/data_loader.py` (see Task)
* **ETT-small/, electricity/, exchange_rate/, illness/, traffic/, weather/** – the raw time series data in `.csv` format (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `electricity`, `exchange_rate`, `national_illness`, `traffic`, `weather`), each up to the start of TSL's test split. You will need these files to correctly load the dataset class in TSL.
* **sample\_predictor/** – a predictor that forecasts 0 everywhere
* **check\_predictor.py**, **predictor\_tools/** – the local check (see Submission)
* **predictor\_datasets.json** – the datasets' shapes, row counts and window counts

## Evaluation

* Your predictor's forecasts will be evaluated based on Mean Squared Error (MSE) between prediction and ground truth at each time stamp, computed separately for each dataset over all of its test windows.
* The ground truth is the target of TSL's data loader for the test split (all variables, as with `--features M`). The loader standardises each dataset with a scaler fitted on its training split, so predict on that standardised scale, not in raw units (do not inverse-transform your predictions).

> ⚠ **Note:** Any attempt to hard-code or replicate the expected results without running the actual model is strictly prohibited.
