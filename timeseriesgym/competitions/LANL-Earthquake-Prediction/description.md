# LANL Earthquake Prediction

## Description
Forecasting earthquakes is one of the most important problems in Earth science because of their devastating consequences. Current scientific studies related to earthquake forecasting focus on three key points: **when** the event will occur, **where** it will occur, and **how large** it will be.

In this competition, you will address **when** the earthquake will take place. Specifically, you'll predict the time remaining before laboratory earthquakes occur from real-time seismic data.

If this challenge is solved and the physics are ultimately shown to scale from the laboratory to the field, researchers will have the potential to improve earthquake hazard assessments that could save lives and billions of dollars in infrastructure.

This challenge is hosted by Los Alamos National Laboratory which enhances national security by ensuring the safety of the U.S. nuclear stockpile, developing technologies to reduce threats from weapons of mass destruction, and solving problems related to energy, environment, infrastructure, health, and global security concerns.

## Acknowledgments
**Geophysics Group:** The competition builds on initial work from Bertrand Rouet-Leduc, Claudia Hulbert, and Paul Johnson. B. Rouet-Leduc prepared the data for the competition.

**Department of Geosciences:** Data are from experiments performed by Chas Bolton, Jacques Riviere, Paul Johnson and Prof. Chris Marone.

**Department of Physics & Astronomy:** This competition stemmed from the DOE Council workshop "Information is in the Noise: Signatures of Evolving Fracture and Fracture Networks" held March 2018 that was organized by Prof. Laura J. Pyrak-Nolte.

**Department of Energy, Office of Science, Basic Energy Sciences, Chemical Sciences, Geosciences and Biosciences Division:** The Geosciences core research.

## Evaluation
Submissions are evaluated using the mean absolute error between the predicted time remaining before the next lab earthquake and the actual remaining time.

In statistics, mean absolute error (MAE) is a measure of errors between paired observations expressing the same phenomenon. MAE is calculated as:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Where:
- $n$ is the number of samples
- $y_i$ is the actual value
- $\hat{y}_i$ is the predicted value

## Submission File
For each `seg_id` in the test set folder, you must predict `time_to_failure`, which is the remaining time before the next lab earthquake. The file should contain a header and have the following format:

```
seg_id,time_to_failure
seg_00030f,0
seg_0012b5,0
seg_00184e,0
...
```

## Prizes
* **1st Place** - $20,000
* **2nd Place** - $15,000
* **3rd Place** - $7,000
* **4th Place** - $5,000
* **5th Place** - $3,000

## Additional Information
The data are from an experiment conducted on rock in a double direct shear geometry subjected to bi-axial loading, a classic laboratory earthquake model.

Two fault gouge layers are sheared simultaneously while subjected to a constant normal load and a prescribed shear velocity. The laboratory faults fail in repetitive cycles of stick and slip that is meant to mimic the cycle of loading and failure on tectonic faults. While the experiment is considerably simpler than a fault in Earth, it shares many physical characteristics.

Los Alamos' initial work showed that the prediction of laboratory earthquakes from continuous seismic data is possible in the case of quasi-periodic laboratory seismic cycles. In this competition, the team has provided a much more challenging dataset with considerably more aperiodic earthquake failures.

## Dataset Description
The goal of this competition is to use seismic signals to predict the timing of laboratory earthquakes. The data comes from a well-known experimental set-up used to study earthquake physics. The `acoustic_data` input signal is used to predict the time remaining before the next laboratory earthquake (`time_to_failure`).

The training data is a single, continuous segment of experimental data. The test data consists of a folder containing many small segments. The data *within* each test file is continuous, but the test files do not represent a continuous segment of the experiment; thus, the predictions cannot be assumed to follow the same regular pattern seen in the training file.

For each `seg_id` in the test folder, you should predict a *single* `time_to_failure` corresponding to the time between the *last row of the segment* and the next laboratory earthquake.

### File descriptions
* **train.csv** - A single, continuous training segment of experimental data.
* **test** - A folder containing many small segments of test data.
* **sample_submission.csv** - A sample submission file in the correct format.

### Data fields
* **acoustic_data** - the seismic signal [int16]
* **time_to_failure** - the time (in seconds) until the next laboratory earthquake [float64]
* **seg_id** - the test segment ids for which predictions should be made (one prediction per segment)
