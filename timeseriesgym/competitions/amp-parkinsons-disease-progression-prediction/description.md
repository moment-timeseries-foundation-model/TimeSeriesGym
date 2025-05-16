# AMP®-Parkinson's Disease Progression Prediction

## Description

### Goal of the Competition
The goal of this competition is to predict MDS-UPDR scores, which measure progression in patients with Parkinson's disease. The Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS) is a comprehensive assessment of both motor and non-motor symptoms associated with Parkinson's. You will develop a model trained on data of protein and peptide levels over time in subjects with Parkinson's disease versus normal age-matched control subjects.

Your work could help provide important breakthrough information about which molecules change as Parkinson's disease progresses.

### Context
Parkinson's disease (PD) is a disabling brain disorder that affects movements, cognition, sleep, and other normal functions. Unfortunately, there is no current cure—and the disease worsens over time. It's estimated that by 2037, 1.6 million people in the U.S. will have Parkinson's disease, at an economic cost approaching $80 billion. Research indicates that protein or peptide abnormalities play a key role in the onset and worsening of this disease. Gaining a better understanding of this—with the help of data science—could provide important clues for the development of new pharmacotherapies to slow the progression or cure Parkinson's disease.

Current efforts have resulted in complex clinical and neurobiological data on over 10,000 subjects for broad sharing with the research community. A number of important findings have been published using this data, but clear biomarkers or cures are still lacking.

Competition host, the Accelerating Medicines Partnership® Parkinson's Disease (AMP®PD), is a public-private partnership between government, industry, and nonprofits that is managed through the Foundation of the National Institutes of Health (FNIH). The Partnership created the AMP PD Knowledge Platform, which includes a deep molecular characterization and longitudinal clinical profiling of Parkinson's disease patients, with the goal of identifying and validating diagnostic, prognostic, and/or disease progression biomarkers for Parkinson's disease.

Your work could help in the search for a cure for Parkinson's disease, which would alleviate the substantial suffering and medical care costs of patients with this disease.

## Dataset Description

The goal of this competition is to predict the course of Parkinson's disease (PD) using protein abundance data. The complete set of proteins involved in PD remains an open research question and any proteins that have predictive value are likely worth investigating further. The core of the dataset consists of protein abundance values derived from mass spectrometry readings of cerebrospinal fluid (CSF) samples gathered from several hundred patients. Each patient contributed several samples over the course of multiple years while they also took assessments of PD severity.

## Files

### train_peptides.csv
Mass spectrometry data at the peptide level. Peptides are the component subunits of proteins.
* `visit_id` - ID code for the visit.
* `visit_month` - The month of the visit, relative to the first visit by the patient.
* `patient_id` - An ID code for the patient.
* `UniProt` - The UniProt ID code for the associated protein. There are often several peptides per protein.
* `Peptide` - The sequence of amino acids included in the peptide. Some rare annotations may not be included in the table. The test set may include peptides not found in the train set.
* `PeptideAbundance` - The frequency of the amino acid in the sample.

### train_proteins.csv
Protein expression frequencies aggregated from the peptide level data.
* `visit_id` - ID code for the visit.
* `visit_month` - The month of the visit, relative to the first visit by the patient.
* `patient_id` - An ID code for the patient.
* `UniProt` - The UniProt ID code for the associated protein. There are often several peptides per protein. The test set may include proteins not found in the train set.
* `NPX` - Normalized protein expression. The frequency of the protein's occurrence in the sample. May not have a 1:1 relationship with the component peptides as some proteins contain repeated copies of a given peptide.

### train_clinical_data.csv
* `visit_id` - ID code for the visit.
* `visit_month` - The month of the visit, relative to the first visit by the patient.
* `patient_id` - An ID code for the patient.
* `updrs_[1-4]` - The patient's score for part N of the Unified Parkinson's Disease Rating Scale. Higher numbers indicate more severe symptoms. Each sub-section covers a distinct category of symptoms, such as mood and behavior for Part 1 and motor functions for Part 3.
* `upd23b_clinical_state_on_medication` - Whether or not the patient was taking medication such as Levodopa during the UPDRS assessment. Expected to mainly affect the scores for Part 3 (motor function). These medications wear off fairly quickly (on the order of one day) so it's common for patients to take the motor function exam twice in a single month, both with and without medication.

### supplemental_clinical_data.csv
Clinical records without any associated CSF samples. This data is intended to provide additional context about the typical progression of Parkinsons. Uses the same columns as **train_clinical_data.csv**.

### test_clinical_data.csv
Contains the same columns as train_clinical_data.csv but for patients in the test set. Used for making predictions.

### test_proteins.csv
Similar to train_proteins.csv but for the test set. May include proteins not found in the training data.

### test_peptides.csv
Similar to train_peptides.csv but for the test set. May include peptides not found in the training data.

### sample_submission.csv
Template file for submitting predictions. You must generate this file.
* `prediction_id` - Identifier code following the format `{patient_id}_{visit_id}_updrs_{updrs_part}_plus_{months}`. Indicates the patient, visit, UPDRS part (1-4), and the number of months in the future for which to predict.
* `rating` - The predicted UPDRS score. This is what you need to fill in for your submission.
* `group_key` - An identifier derived from the patient_id portion of the prediction_id, useful for grouping related predictions.

## Evaluation
Submissions are evaluated on SMAPE between forecasts and actual values. We define SMAPE = 0 when the actual and predicted values are both 0.

For each patient visit where a protein/peptide sample was taken you will need to estimate both their UPDRS scores for that visit and predict their scores for any potential visits 6, 12, and 24 months later. Predictions for any visits that didn't ultimately take place are ignored.

The **Symmetric Mean Absolute Percentage Error** (**SMAPE** or **sMAPE**) is an accuracy measure based on percentage (or relative) errors. It is usually defined as follows:

$$\text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \frac{|F_t - A_t|}{(|A_t| + |F_t|)/2}$$

where:
- *A<sub>t</sub>* is the actual value
- *F<sub>t</sub>* is the forecast value

The absolute difference between *A<sub>t</sub>* and *F<sub>t</sub>* is divided by half the sum of absolute values of the actual value *A<sub>t</sub>* and the forecast value *F<sub>t</sub>*. The value of this calculation is summed for every fitted point *t* and divided again by the number of fitted points *n*.

## Prizes
- 1st Place - $25,000
- 2nd Place - $20,000
- 3rd Place - $15,000

## Citation
Leslie Kirsch, Sohier Dane, Stacey Adam, and Victoria Dardov. AMP®-Parkinson's Disease Progression Prediction. https://kaggle.com/competitions/amp-parkinsons-disease-progression-prediction, 2023. Kaggle.

## Changelog
Removed the timeline and code requirements.
