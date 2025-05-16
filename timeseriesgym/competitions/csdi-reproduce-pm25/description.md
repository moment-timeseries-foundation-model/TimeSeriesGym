# Time Series Imputation Challenge with CSDI

## Challenge Overview

This challenge focuses on reproducing and optimizing results for time series imputation using a specific research method. You will work with a pre-existing research codebase and a real-world dataset to train, evaluate, and optimize a state-of-the-art time series imputation model.

## Description

Time series imputation (filling in missing values in sequential data) is critical in domains like healthcare monitoring, financial forecasting, and environmental sensing. In this challenge, you will implement and optimize [Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation (CSDI)](https://arxiv.org/abs/2107.03502), a powerful approach that leverages diffusion models to generate realistic imputations while accounting for uncertainty.

Your primary objective is to:
1. Understand the CSDI implementation
2. Train the model on the PM 2.5 Air Quality dataset
3. Optimize the model configuration to achieve the best possible imputation performance

## Detailed Tasks

### 1. Environment Setup (Required)
- Navigate to the provided CSDI repository located in your input data directory
- Install all dependencies specified in `requirements.txt`
- Download and prepare the PM 2.5 Air Quality dataset

### 2. Code Understanding (Required)
- Review the repository structure, especially the main execution scripts (`exe_pm25.py`) and configuration files (`config/base.yaml`)
- Understand how the model is trained, how imputations are generated, and how performance is evaluated

### 3. Experimentation (Required)
- Train the CSDI model on the PM 2.5 dataset
- Generate imputations and evaluate performance
- **IMPORTANT**: Save all results to the submission directory
  - The trained model weights
  - The evaluation metrics file (named `result_nsample{N}.pk`, where `{N}` represents the number of samples to generate during evaluation, e.g., `result_nsample10.pk`)

```bash
# Example command (verify this matches the repository instructions)
python exe_pm25.py --nsample 10  # Adjust parameters as needed
```

### 4. Optimization (Recommended)
- Modify the configuration (`config/base.yaml`) to improve performance
- Consider adjusting:
  - Network architecture (layers, hidden dimensions)
  - Training parameters (learning rate, batch size)
  - Diffusion process parameters (noise schedule, number of steps)
- Run multiple experiments with different configurations
- Select and submit your best-performing model

## Data Description

The PM 2.5 Air Quality dataset contains measurements of fine particulate matter (PM 2.5) from multiple monitoring stations over time. It features:
- Time-stamped measurements
- Multiple locations (spatial dimension)
- Natural missing values
- Temporal patterns and correlations

The dataset is referenced in: [ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data](https://www.microsoft.com/en-us/research/publication/st-mvl-filling-missing-values-in-geo-sensory-time-series-data/)

## Computational Considerations

CSDI uses diffusion models, which can be computationally intensive to train. To manage resources effectively:

- Start with a reduced configuration (fewer epochs, larger batch size, smaller model)
- Use early experiments to verify your setup and workflow
- Progressively refine your approach as you understand the model behavior
- Monitor training time and resource usage to ensure completion within the challenge timeframe
- Consider checkpointing to save intermediate progress

## Evaluation Criteria

Your submission will be evaluated based on the Mean Absolute Error (MAE) of your imputation results. While the output file (`result_nsample{N}.pk`) contains multiple metrics including RMSE and CRPS, **only the MAE will be used for the final ranking**.

A valid submission **must include**:
1. The trained model weights
2. The results file (`result_nsample{N}.pk`, where `{N}` is the number of samples used for evaluation) in the submission directory

## Important Notes

- **Validation**: Your code must actually train the model and generate results. Hard-coding or copying expected outputs is prohibited and will result in disqualification.
- **Reproducibility**: Set random seeds if you want consistent results across runs.
- **Documentation**: Include a brief description of any modifications you made to the original code or configuration.
- **Time Management**: Allow sufficient time for model training, as diffusion models can be computationally intensive.

Good luck!
