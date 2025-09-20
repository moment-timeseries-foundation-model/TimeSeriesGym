# OSIC Pulmonary Fibrosis Progression

## Overview

### Description

Imagine one day, your breathing became consistently labored and shallow. Months later you were finally diagnosed with pulmonary fibrosis, a disorder with no known cause and no known cure, created by scarring of the lungs. If that happened to you, you would want to know your prognosis. That’s where a troubling disease becomes frightening for the patient: outcomes can range from long-term stability to rapid deterioration, but doctors aren’t easily able to tell where an individual may fall on that spectrum. Your help, and data science, may be able to aid in this prediction, which would dramatically help both patients and clinicians.

Current methods make fibrotic lung diseases difficult to treat, even with access to a chest CT scan. In addition, the wide range of varied prognoses create issues organizing clinical trials. Finally, patients suffer extreme anxiety—in addition to fibrosis-related symptoms—from the disease’s opaque path of progression.

[Open Source Imaging Consortium (OSIC)](https://www.osicild.org/) is a not-for-profit, co-operative effort between academia, industry and philanthropy. The group enables rapid advances in the fight against Idiopathic Pulmonary Fibrosis (IPF), fibrosing interstitial lung diseases (ILDs), and other respiratory diseases, including emphysematous conditions. Its mission is to bring together radiologists, clinicians and computational scientists from around the world to improve imaging-based treatments.

In this competition, you’ll predict a patient’s severity of decline in lung function based on a CT scan of their lungs. You’ll determine lung function based on output from a spirometer, which measures the volume of air inhaled and exhaled. The challenge is to use machine learning techniques to make a prediction with the image, metadata, and baseline FVC as input.

If successful, patients and their families would better understand their prognosis when they are first diagnosed with this incurable lung disease. Improved severity detection would also positively impact treatment trial design and accelerate the clinical development of novel treatments.

**This is a Code Competition. Refer to Code Requirements for details.**

### Evaluation

This competition is evaluated on a modified version of the Laplace Log Likelihood. In medical applications, it is useful to evaluate a model's confidence in its decisions. Accordingly, the metric is designed to reflect both the accuracy and certainty of each prediction.

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation $\sigma$). The metric is computed as:

$$\sigma_{\text{clipped}} = \max(\sigma, 70)$$
$$\Delta = \min(|\text{FVC}_{\text{true}} - \text{FVC}_{\text{predicted}}|, 1000)$$
$$\text{metric} = - \frac{\sqrt{2} \Delta}{\sigma_{\text{clipped}}} - \ln(\sqrt{2} \sigma_{\text{clipped}})$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). Note that metric values will be negative and higher is better.

### Submission File

For each `Patient_Week`, you must predict the `FVC` and a confidence. To avoid potential leakage in the timing of follow up visits, you are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.
```

### Prizes

- 1st Place - $30,000  
- 2nd Place - $15,000  
- 3rd Place - $10,000  

### Code Requirements
**This is a Code Competition**

Submissions to this competition must be made through Notebooks. In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 4 hours run-time
- TPUs will not be available for making submissions to this competition. You are still welcome to use them for training models.
- No internet access enabled
- External data, freely & publicly available, is allowed. This includes pre-trained models.
- Submission file must be named `submission.csv`

Please see the Code Competition FAQ for more information on how to submit.

## Dataset Description

The aim of this competition is to predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.  
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

There are around 200 cases in the public & private test sets, combined. This is split roughly 15-85 between public-private.

Since this is real medical data, you will notice the relative timing of `FVC` measurements varies widely. The timing of the initial measurement relative to the CT scan and the duration to the forecasted time points may be different for each patient. This is considered part of the challenge of the competition. To avoid potential leakage in the timing of follow up visits, you are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

### Files

This is a synchronous rerun code competition. The provided test set is a small representative set of files (copied from the training set) to demonstrate the format of the private test set. When you submit your notebook, Kaggle will rerun your code on the test set, which contains unseen images.

- **train.csv** - the training set, contains full history of clinical information  
- **test.csv** - the test set, contains only the baseline measurement  
- **train/** - contains the training patients' baseline CT scan in DICOM format  
- **test/** - contains the test patients' baseline CT scan in DICOM format  
- **sample_submission.csv** - demonstrates the submission format  

### Columns

**train.csv and test.csv**

- `Patient` - a unique Id for each patient (also the name of the patient's DICOM folder)  
- `Weeks` - the relative number of weeks pre/post the baseline CT (may be negative)  
- `FVC` - the recorded lung capacity in ml  
- `Percent` - a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics  
- `Age`  
- `Sex`  
- `SmokingStatus`  

**sample_submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)  
- `FVC` - the predicted FVC in ml  
- `Confidence` - a confidence value of your prediction (also has units of ml)  

## Citation

Ahmed Shahin, Carmela Wegworth, David, Elizabeth Estes, Julia Elliott, Justin Zita, SimonWalsh, Slepetys, and Will Cukierski. OSIC Pulmonary Fibrosis Progression. https://kaggle.com/competitions/osic-pulmonary-fibrosis-progression, 2020. Kaggle.