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
- Download and prepare the PM 2.5 Air Quality dataset


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
- Generate imputations and evaluate performance
- **IMPORTANT**: Save all results to the submission directory
  - The trained model weights
  - The evaluation metrics file (named `result_nsample{N}.pk`, where `{N}` represents the number of samples generated during evaluation, e.g., `result_nsample10.pk`)

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

Your submission will be evaluated based on:
1. **Correctness of implementation**: Your `CSDIModelPM25` class must correctly process and handle the PM 2.5 dataset
2. **Performance**: The Mean Absolute Error (MAE) of your model's imputations

While the output file (`result_nsample{N}.pk`) contains multiple metrics including RMSE and CRPS, **only the MAE will be used for the final ranking**.

A valid submission **must include**:
1. Your completed implementation of `CSDIModelPM25` in `main_model.py`
2. The trained model weights
3. The results file (`result_nsample{N}.pk`, where `{N}` is the number of samples used for evaluation) in the submission directory

## Important Notes

- **Validation**: Your code must actually train the model and generate results. Hard-coding or copying expected outputs is prohibited and will result in disqualification.
- **Reproducibility**: Set random seeds if you want consistent results across runs.
- **Documentation**: Include a brief description of your implementation approach and any modifications you made to the original code or configuration.
- **Time Management**: Allow sufficient time for model training, as diffusion models can be computationally intensive.
- **Code Quality**: Your implementation should be well-structured, properly commented, and follow good programming practices.

Good luck!
