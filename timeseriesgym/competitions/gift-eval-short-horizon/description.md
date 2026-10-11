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
