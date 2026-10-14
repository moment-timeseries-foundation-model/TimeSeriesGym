# Time Series Imputation Challenge with CSDI: Implementation and Optimization

## Challenge Overview

This challenge focuses on implementing, training, and optimizing a Conditional Score-based Diffusion Model for time series imputation. You will work with a partially implemented research codebase and the PM 2.5 Air Quality dataset to complete the model implementation, train it, and optimize its performance for missing value imputation.

## Description

Time series imputation (filling in missing values in sequential data) is critical in domains like healthcare monitoring, financial forecasting, and environmental sensing. In this challenge, you will:

1. Complete the implementation of [Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation (CSDI)](https://arxiv.org/abs/2107.03502) for the PM 2.5 Air Quality dataset
2. Train the completed model implementation
3. Optimize the model configuration to achieve the best possible imputation performance

This challenge tests both your ability to understand and implement machine learning research code and your skill in optimizing model performance for real-world data.

## Detailed Tasks

### 1. Environment Setup (Required)
- Navigate to the provided CSDI repository located in your input data directory
- Install all dependencies specified in `requirements.txt`
- The PM 2.5 Air Quality dataset is already prepared in the repository's `data/pm25/` folder, in the layout that `python download.py pm25` creates (see "Data Description"). Do not run `download.py pm25`: it would replace the prepared files


### 2. Code Understanding (Required)
- Review the repository structure, especially the main execution scripts (`exe_pm25.py`) and configuration files (`config/base.yaml`).
- Pay special attention to `main_model.py` which contains the incomplete `CSDIModelPM25` class
- Study the `CSDIModelPhysio` class implementation as a reference for your own implementation

### 3. Code Implementation (Required)
- Complete the implementation of the `CSDIModelPM25` class in `main_model.py`
- Key implementation tasks:
  - Implement data processing logic specific to PM2.5 data
  - Ensure proper handling of the dataset's format, dimensions, and characteristics
  - Implement any PM2.5-specific preprocessing steps
  - Use `CSDIModelPhysio` as a reference, but adapt it for the PM2.5 dataset's unique properties
- Verify your implementation by running basic tests before proceeding to full training

### 4. Experimentation (Required)
- Train your implemented CSDI model on the PM 2.5 dataset
- Use your trained model to impute the evaluation targets of the test months (listed in `sample_submission.csv`)
- **IMPORTANT**: Save your results to the submission directory
  - `submission.csv` with your imputed values (required; see "Evaluation Criteria")
  - Your completed `main_model.py`
  - Optionally, the trained model weights

```bash
# Example command
python exe_pm25.py --nsample 10  # Adjust parameters as needed
```

### 5. Optimization (Recommended)
- Modify the configuration (`config/base.yaml`) to improve performance
- Consider adjusting:
  - Network architecture (layers, hidden dimensions)
  - Training parameters (learning rate, batch size)
  - Diffusion process parameters (noise schedule, number of steps)
  - Data preprocessing parameters
- Run multiple experiments with different configurations
- Select and submit your best-performing model

## Data Description

The PM 2.5 Air Quality dataset contains measurements of fine particulate matter (PM 2.5) from multiple monitoring stations over time. It features:
- Time-stamped measurements across 36 monitoring stations in Beijing
- Hourly readings over multiple months
- Natural missing values due to sensor failures or maintenance
- Temporal patterns (daily and seasonal cycles) and spatial correlations
- Different characteristics from the healthcare dataset used in `CSDIModelPhysio`

The dataset is referenced in: [ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data](https://www.microsoft.com/en-us/research/publication/st-mvl-filling-missing-values-in-geo-sensory-time-series-data/)

The data are hourly readings from May 2014 to April 2015, in the files that CSDI's loader (`dataset_pm25.py`) reads:
- `data/pm25/Code/STMVL/SampleData/pm25_ground.txt`: the readings, one column per station; an empty field is a missing value
- `data/pm25/Code/STMVL/SampleData/pm25_missing.txt`: the same readings with additional values removed; CSDI imputes these removed values and scores its imputations on them
- `data/pm25/Code/STMVL/SampleData/pm25_latlng.txt`: the stations' coordinates
- `data/pm25/pm25_meanstd.pk`: the per-station mean and standard deviation that CSDI normalizes with, computed from the months that are not test months

CSDI tests on March, June, September and December, and trains and validates on the other months. **The true values of the test months' evaluation targets are withheld**: in the provided `pm25_ground.txt` they are empty, so in the test months it has the same values as `pm25_missing.txt`. The other months are as published.

## Implementation Hints

When implementing `CSDIModelPM25`, consider these tips:
- Carefully analyze the data format and structure of the PM2.5 dataset
- Identify how the PM2.5 dataset differs from the PhysioNet dataset used in `CSDIModelPhysio`
- Ensure your implementation handles the correct number of variables (monitoring stations)
- Adapt the temporal encoding to match the PM2.5 dataset's time structure
- Implement appropriate data normalization for air quality measurements
- Consider the spatial relationships between monitoring stations

## Computational Considerations

CSDI uses diffusion models, which can be computationally intensive to train. To manage resources effectively:
- Start with a reduced configuration (fewer epochs, larger batch size, smaller model)
- Use early experiments to verify your implementation is correct
- Progressively refine your approach as you understand the model behavior
- Monitor training time and resource usage to ensure completion within the challenge timeframe
- Consider checkpointing to save intermediate progress

## Evaluation Criteria

Your imputations must come from your implementation of `CSDIModelPM25`, and they will be evaluated based on the Mean Absolute Error (MAE), in the original PM 2.5 units, of your imputed values at the evaluation targets of the test months: the entries that are observed in the original `pm25_ground.txt` but missing in `pm25_missing.txt`. These are the entries and the metric of CSDI's own test evaluation (`evaluate` in `utils.py`, which takes the median of the generated samples as the imputation). Lower is better.

Since the true values are withheld, the test evaluation at the end of `exe_pm25.py` has nothing to score: it finds no evaluation targets and stops with a division by zero, after the trained weights have been saved (`model.pth` in the run's folder under `save/`). Write the imputations yourself: impute the test windows (`PM25Dataset(mode="test")`) with the trained model, for example with `model.evaluate(batch, nsample)` and the median of the samples, convert them back to the original units (times `train_std`, plus `train_mean`), and write the value of each target listed in `sample_submission.csv`. The target mask that `model.evaluate` returns (its third value, `eval_points` in `evaluate` in `utils.py`) is empty here, all zeros: it marks the entries observed in `pm25_ground.txt` but missing in `pm25_missing.txt`, and with the targets withheld the two files agree in the test months. Do not select the values to submit with it; select them by the ids in `sample_submission.csv`, each of which names a time and a station (a row and a column of `pm25_ground.txt`).

A valid submission **must include** `submission.csv` in the submission directory, with the columns `id,value`:
- exactly the `id`s of `sample_submission.csv`, in the same order, each once
- `id` is `<datetime>_<station>`: the time as written in `pm25_ground.txt` and the station's column name, e.g. `2014/06/01 01:00:00_001001`
- `value` is your imputed PM 2.5 value, a finite number

A submission with missing, extra, repeated or reordered `id`s, or with a value that is not a finite number, is invalid. Also save your completed `main_model.py` in the submission directory. It, the trained model weights and the `result_nsample{N}.pk` file are not scored.

## Important Notes

- **Validation**: Your code must actually train the model and generate results. Hard-coding or copying expected outputs is prohibited and will result in disqualification. This includes the withheld values: obtaining the original STMVL files, or using them in any way, is prohibited.
- **Reproducibility**: Set random seeds if you want consistent results across runs.
- **Documentation**: Include a brief description of your implementation approach and any modifications you made to the original code or configuration.
- **Time Management**: Allow sufficient time for model training, as diffusion models can be computationally intensive.
- **Code Quality**: Your implementation should be well-structured, properly commented, and follow good programming practices.

Good luck!
