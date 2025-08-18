# Overview

## Description

In this challenge, you will be evaluated on your ability to translate state-of-the-art research into functional code implementations. We will implement a state-of-the-art deep learning model, [iTransformer](https://arxiv.org/pdf/2310.06625), for time series forecasting using the Time Series Library (TSL) framework. Once implemented, we will validate the implementation by forecasting on a weather dataset containing 21 meteorological variables, recorded every 10 minutes at the Max Planck Biogeochemistry Institute in 2020.

Your tasks are:

1. Complete the partial implementation of iTransformer.
2. Train and evaluate the model using a specified configuration.

## Task

1. **Install the Time Series Library (TSL)**

   * The TSL repository is located in your input data directory.
   * To install dependencies:

   ```bash
   cd Time-Series-Library
   pip install -r requirements.txt
   ```

2. **Complete the iTransformer implementation**

   * Open `Time-Series-Library/models/iTransformer.py`.
   * Implement the missing sections marked with `TODO`.
   * We’ve added the pertinent sections of the original iTransformer paper to the file `itransformer.txt` in the input directory.

3. **Run the evaluation**

   * Use `run.py` to train and evaluate your model with the following settings:

     * Encoder layers: 1
     * Decoder layers: 1
     * Lookback window: 96
     * Forecast horizon: 96
     * Encoder input size: 21
     * Decoder input size: 21
     * Output size: 21
     * Model dimension: 64
     * Feedforward dimension: 64
     * Training epochs: 1
   * The raw dataset `weather.csv` is provided in the `data` directory. You will need to configure the data loader accordingly.
   * Set an output path to save the experiment results.
   * **Hint:** Check `run.py` for available command-line arguments. Example scripts in the `scripts/` folder may also be helpful.

## Data

Your `data` directory includes the following:

* **Time-Series-Library/** – the cloned TSL repository
* **weather.csv** – the raw weather time series data
* **itransformer.pdf** – reference paper for iTransformer implementation
* **sample\_submission.npy** – an example output; your pipeline will generate this automatically if configured correctly

## Evaluation

* If implemented correctly and the output path is set to the `submission/` folder, your pipeline will generate a `pred.npy` file containing your model's predictions.
* **Score = 100** if your results exactly match the reference.
* **Score = 0** if the file is missing, incorrect, or manually fabricated.

> ⚠ **Note:** Any attempt to hard-code or replicate the expected results without running the actual model will be considered an academic integrity violation.
