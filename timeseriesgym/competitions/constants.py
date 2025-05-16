# These competitions can be directly downloaded via the Kaggle API
kaggle_original_competitions = [
    "amp-parkinsons-disease-progression-prediction",
    "ashrae-energy-prediction",
    "child-mind-institute-detect-sleep-states",
    "g2net-gravitational-wave-detection",
    "LANL-Earthquake-Prediction",
    "m5-forecasting-accuracy",
    "new-york-city-taxi-fare-prediction",
    "online-sales",
    "optiver-realized-volatility-prediction",
    "recruit-restaurant-visitor-forecasting",
    "sberbank-russian-housing-market",
    "spaceship-titanic",
    "ventilator-pressure-prediction",
    "g2net-gravitational-wave-detection",
    "amp-parkinsons-disease-progression-prediction",
    "hms-harmful-brain-activity-classification",
]

# These competitions are not available on Kaggle but must be downloaded from other sources
non_kaggle_original_competitions = [
    "context-is-key-moirai",
    "context-is-key-nn5-chronos",
    "gift-eval-short-horizon",
    "MIT-BIH-Arrhythmia",
    "moment-anomaly-detection-score",
    "ptb-xl-classification-challenge",
    "time-series-exam-1-1",
    "time-series-library-itransformer-weather-forecast",
    "time-series-library-long-horizon-forecast",
    "ucr-anomaly-detection",
]

# These competitions are derived from the other competitions
derived_competitions = [
    "MIT-BIH-Arrhythmia-Weak-Supervision",
    "ventilator-pressure-prediction-missing",
    "optiver-realized-volatility-prediction-missing",
    "optiver-realized-volatility-prediction-hyperparameter-search",
    "ptb-xl-classification-challenge-hyperparameter-search",
    "ptb-xl-classification-challenge-feature-enhancement",
]

# These competitions are coding competitions and require the agent to submit code file(s)
# They typically do not require any data in the form of zip files to be downloaded
coding_competitions = [
    "csdi-reproduce-pm25",
    "csdi-implement-model-pm25",
    "resnet-tensorflow-to-pytorch",
    "stomp-R-to-python",
]
