# MIT-BIH Arrhythmia Detection

The following is taken from: https://physionet.org/content/mitdb/1.0.0/

## Background
Since 1975, our laboratories at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center) and at MIT have supported our own research into arrhythmia analysis and related subjects. One of the first major products of that effort was the MIT-BIH Arrhythmia Database, which we completed and began distributing in 1980. The database was the first generally available set of standard test material for evaluation of arrhythmia detectors, and has been used for that purpose as well as for basic research into cardiac dynamics at more than 500 sites worldwide.

In this competition, you'll tackle the critical challenge of **what** type of cardiac rhythm is present in electrocardiogram (ECG) data. Specifically, you'll classify heartbeats into six distinct categories from the gold-standard MIT-BIH Arrhythmia Database, including normal beats and five types of arrhythmias.

Heart disease remains the leading cause of death globally, claiming nearly 18 million lives each year. Accurate automatic detection of cardiac arrhythmias from ECG signals can dramatically improve early diagnosis, enabling timely intervention and potentially saving millions of lives.

## The Challenge

Your task is to develop an algorithm that can accurately classify each heartbeat from ECG signals into one of six classes:
- Normal beats
- Premature Ventricular Contractions (PVCs)
- Paced beats
- Atrial premature beats
- Fusion beats
- Noise

## Data Description
The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Twenty-three recordings were chosen at random from a set of 4000 24-hour ambulatory ECG recordings collected from a mixed population of inpatients (about 60%) and outpatients (about 40%) at Boston's Beth Israel Hospital; the remaining 25 recordings were selected from the same set to include less common but clinically significant arrhythmias that would not be well-represented in a small random sample.

The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range. Two or more cardiologists independently annotated each record; disagreements were resolved to obtain the computer-readable reference annotations for each beat (approximately 110,000 annotations in all) included with the database.

This directory contains the entire MIT-BIH Arrhythmia Database. About half (25 of 48 complete records, and reference annotation files for all 48 records) of this database has been freely available here since PhysioNet's inception in September 1999. The 23 remaining signal files, which had been available only on the MIT-BIH Arrhythmia Database CD-ROM, were posted here in February 2005.

## File Descriptions

* **train_features.h5** - Contains the ECG signal segments for training. The file has a group structure with a 'train' group containing two leads ("lead_1" and "lead_2") as datasets. In most records, the upper signal is a modified limb lead II (MLII) and the lower signal is usually a modified lead V1. Each lead contains ECG signal segments that have been preprocessed and normalized. Access the data with `h5py` using the path structure 'train/LEAD_NUMBER' (e.g., 'train/lead_1', 'train/lead_2').

* **train_labels.h5** - Contains one-hot encoded labels for the training dataset under the "labels" group. Each label corresponds to one of six classification categories: Normal (N), Premature Ventricular Contraction (V), Paced Beat (/), Atrial Premature Beat (A), Fusion Beat (F), and Noise (~).

* **test_features.h5** - Contains the ECG signal segments for testing. It follows the same structure as the training features with a 'test' group containing two leads ("lead_1" and "lead_2") as datasets. Each lead contains ECG signal segments that have been preprocessed and normalized. These are the beats you need to classify.

* **sample_submission.h5** - Template submission file in the required format. Contains a dataset named 'submission' with dimensions (N_test, 6), where each row represents a one-hot encoded prediction across the six classes for a test beat. Use this as a reference for formatting your submissions.

### Loading the Data

Here's a code snippet to help you get started with loading the data:

```python
import h5py
import numpy as np

# Load training data
with h5py.File('train_features.h5', 'r') as f:
    # List available leads
    print(f['train'].keys())

    # Load specific leads
    lead_1_train = f['train/lead_1'][()]
    lead_2_train = f['train/lead_2'][()]

# Load training labels
with h5py.File('train_labels.h5', 'r') as f:
    labels = f['labels'][()]

# Load test data
with h5py.File('test_features.h5', 'r') as f:
    lead_1_test = f['test/lead_1'][()]
    lead_2_test = f['test/lead_2'][()]

# Check shapes
print(f"Lead 1 training data shape: {mlii_train.shape}")
print(f"Training labels shape: {mlii_labels.shape}")
```

### Class Mapping

The six classes in the dataset are:
1. Normal Beat (N)
2. Premature Ventricular Contraction (V)
3. Paced Beat (/)
4. Atrial Premature Beat (A)
5. Fusion Beat (F)
6. Noise (~)

The one-hot encoded labels follow this order. For example, [1,0,0,0,0,0] represents a Normal Beat, while [0,1,0,0,0,0] represents a Premature Ventricular Contraction.

### Selection criteria
The source of the ECGs included in the MIT-BIH Arrhythmia Database is a set of over 4000 long-term Holter recordings that were obtained by the Beth Israel Hospital Arrhythmia Laboratory between 1975 and 1979. Approximately 60% of these recordings were obtained from inpatients. The database contains 23 records (numbered from 100 to 124 inclusive with some numbers missing) chosen at random from this set, and 25 records (numbered from 200 to 234 inclusive, again with some numbers missing) selected from the same set to include a variety of rare but clinically important phenomena that would not be well-represented by a small random sample of Holter recordings. Each of the 48 records is slightly over 30 minutes long.

### ECG lead configuration
In most records, the upper signal is a modified limb lead II (MLII), obtained by placing the electrodes on the chest. The lower signal is usually a modified lead V1 (occasionally V2 or V5, and in one instance V4); as for the upper signal, the electrodes are also placed on the chest. This configuration is routinely used by the BIH Arrhythmia Laboratory. Normal QRS complexes are usually prominent in the upper signal. The lead axis for the lower signal may be nearly orthogonal to the mean cardiac electrical axis, however (i.e., normal beats are usually biphasic and may be nearly isoelectric). Thus normal beats are frequently difficult to discern in the lower signal, although ectopic beats will often be more prominent.


## Evaluation
Submissions will be evaluated using classification accuracy.


## Submission Format
Each submission must be in HDF5 format (`.h5` file) containing your predictions as a matrix. If there are N beats in the test set, your submission should be a matrix of shape (N, 6), where each row represents which is a one-hot representation of the predicted label.

### Requirements
- File format: HDF5 (.h5)
- Dataset name within the file: labels
- Matrix shape: (N, 6) where N is the number of test beats
- Values: 0 or 1. Each row must sum to 1

### Example
Here's how to structure your submission file:

```python
import h5py
import numpy as np

# Your model's predictions for each beat
# Shape: (N, 6) where N is the number of test beats
predictions = np.array([
    [1, 0, 0, 0, 0, 0],  # Beat 1: likely Normal
    [0, 1, 0, 0, 0, 0],  # Beat 2: likely PVC
    [0, 0, 1, 0, 0, 0],  # Beat 3: likely Paced
    # ... more beats
])

# Save predictions to HDF5 file
with h5py.File('submission.h5', 'w') as f:
    f.create_dataset('submission', data=predictions)
```

## References

When using this resource, please cite the original publication:
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)

Please include the standard citation for PhysioNet:
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

### Additional References
1. Mark RG, Schluter PS, Moody GB, Devlin, PH, Chernoff, D. An annotated ECG database for evaluating arrhythmia detectors. IEEE Transactions on Biomedical Engineering 29(8):600 (1982).
2. Moody GB, Mark RG. The MIT-BIH Arrhythmia Database on CD-ROM and software for use with it. Computers in Cardiology 17:185-188 (1990).
