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
   * Set an output path to save the experiment results.
   * **Hint:** Check `run.py` for available command-line arguments. Example scripts in the `scripts/` folder may also be helpful.

## Submission
   * If you utilize `run.py` correctly, the prediction will be saved to your specified output path as a .npy file in shape of `(num_samples, prediction_len, num_variables)`
   * You are required to save the prediction in a single `.h5` file that will be loaded as a dictionary using following code
   ```python
   def load_arbitrary_h5(root):
      if isinstance(root, h5py.Dataset):
         return root[()]

      root_dict = {}
      stack = [(root, root_dict)]

      while stack:
         group, output_dict = stack.pop()

         for name, child in group.items():
            if isinstance(child, h5py.Dataset):
                  output_dict[name] = child[()]
            else:
                  subgroup_dict = {}
                  output_dict[name] = subgroup_dict
                  stack.append((child, subgroup_dict))

      return root_dict

   output_h5_file = ... #submission file path
   with h5py.File(output_h5_file, "r") as answers_file:
      submission = load_arbitrary_h5(answers_file["labels"])
   ```
   * Each dataset under the `labels` group must use the **competition name** as the key and contain a **prediction array with the required shape** as the value.
   * Submissions that do not follow this format will receive a score of **0**.

## Data

Your `data` directory includes the following:

* **Time-Series-Library/** – the cloned TSL repository
* **datanames/** – the raw time series data in `.csv` format. You will need these file to correctly load the dataset class in TSL.
* **sample\_submission.h5** – an example output for submission

## Evaluation

* Your prediction will be evaluated base on Mean Squared Error (MSE) between prediction and ground truth at each time stamp.

> ⚠ **Note:** Any attempt to hard-code or replicate the expected results without running the actual model is strictly prohibited.
