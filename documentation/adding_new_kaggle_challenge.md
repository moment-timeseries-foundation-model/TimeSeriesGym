# Adding a New Competition to TimeSeriesGym

This guide walks through the process of adding a new Kaggle competition to TimeSeriesGym. Follow these steps to ensure your competition is properly integrated with all necessary components.

## Overview

Adding a new competition requires several components:

1. Competition directory structure
2. Data preparation scripts
3. Grading functions
4. Configuration files
5. Documentation
6. Unit tests

## Step 1: Create Competition Directory

Create a new directory for your competition in the TimeSeriesGym structure:

```bash
mkdir -p timeseriesgym/competitions/<competition-id>
```

> **Important**: The directory name must match the competition ID exactly. You can find the competition ID from the Kaggle dataset download command:
>
> ```bash
> kaggle competitions download -c <competition-id>
> ```
>
> For example, for the LANL Earthquake Prediction challenge, the competition ID is `LANL-Earthquake-Prediction`.

After creating the directory, add the competition ID to the appropriate list in `constants.py`:

**Quick Download Tip**:
You can use the TimeSeriesGym CLI to download the competition data:

```bash
timeseriesgym prepare -c <competition-id> --keep-raw
```

Note that this command will throw errors if you haven't completed Step 2, but it's useful to have the data downloaded for reference.

## Step 2: Create Required Files

### 2.1 Competition Description (`description.md`)

Create a `description.md` file that explains the competition. Since there's no direct way to download this from Kaggle, you can use an LLM to help generate the markdown from the Kaggle description.

**Suggested Prompt for LLMs**:
```
Here is the description of a Kaggle Competition. Help me convert it to markdown. Make sure to remove unnecessary information (e.g., links) which are artifacts of copying.

Competition name: <Name of the Competition>
Link: https://www.kaggle.com/competitions/<competition-slug>/overview

Here is the copied description:
<Copy of the Overview>
<Copy of the Dataset Description>
```

Guidelines for content:
- **Include**: Overview, dataset description, citations, prize money, code requirements, acknowledgements
- **Exclude**: Timeline and other non-essential information

### 2.2 Data Preparation Script (`prepare.py`)

Create a `prepare.py` file containing a `prepare` function that processes the raw Kaggle data into the TimeSeriesGym format:

```python
import shutil
from pathlib import Path
import pandas as pd
# Additional imports as needed

def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Process the raw competition files into public (training) and private (testing) directories.

    Args:
        raw: Path to the directory containing raw downloaded files
        public: Path to the directory where public (training) files should be placed
        private: Path to the directory where private (testing) files should be placed
    """
    # Copy/process training data to public folder
    # Example:
    train_data = pd.read_csv(raw / "train.csv")
    train_data.to_csv(public / "train.csv", index=False)

    # Copy/process test data to private folder
    # Example:
    test_data = pd.read_csv(raw / "test.csv")
    test_data.to_csv(private / "test.csv", index=False)

    # IMPORTANT: Copy sample submission to public folder
    shutil.copy(raw / "sample_submission.csv", public / "sample_submission.csv")
```

**Important Guidelines**:
- The `raw` directory contains the extracted files from the Kaggle download
- The `public` directory should contain all files needed for training
- The `private` directory should contain all files needed for testing/submission
- Always include the sample submission file in the `public` directory

### 2.3 Grading Script (`grade.py`)

Create a `grade.py` file with a `grade` function that evaluates submissions according to the competition's scoring metric:

```python
import pandas as pd
import numpy as np
# Additional imports as needed

def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Evaluate a submission against the ground truth answers.

    Args:
        submission: DataFrame containing the predictions
        answers: DataFrame containing the ground truth

    Returns:
        float: Score value (higher is better, unless specified otherwise in config.yaml)
    """
    # Implement the competition's scoring metric
    # Example for Mean Squared Error:
    mse = ((submission['prediction'] - answers['target']).pow(2)).mean()

    # Return the score (note: for metrics where lower is better, you may need to negate)
    return -mse  # Negative since lower MSE is better but higher score is better
```

Make sure your grading function implements the exact same metric used by the Kaggle competition.

### 2.4 Leaderboard File (`leaderboard.csv`)

Create an empty `leaderboard.csv` file, which will be populated automatically:

```bash
touch timeseriesgym/competitions/<competition-id>/leaderboard.csv
```

Then download the leaderboard data:

```bash
timeseriesgym dev download-leaderboard -c <competition-id> --force
```

### 2.5 Configuration File (`config.yaml`)

The `config.yaml` file is crucial for defining how your competition functions within the TimeSeriesGym framework. Different competition types require different configuration parameters. Below is a comprehensive template with detailed comments:

```yaml
# -----------------------------------------------------------------------------------
# Basic Competition Information
# -----------------------------------------------------------------------------------
# Unique identifier for the competition - use kebab-case (required)
id: <competition-id>

# Human-readable name of the competition (required)
name: <Full Competition Name>

# Identifier for the competition on Kaggle (null for non-Kaggle competitions)
kaggle_competition_id: null

# Competition type (one of: simple, classification, regression, clustering, time_series, etc.)
# This affects how the competition is categorized and potentially how it's graded
competition_type: simple

# Whether the competition is significant enough to award medals (true/false)
awards_medals: false

# Optional: Parent competition ID if this is a derivative of another competition
# parent_id: <parent-competition-id>

# Optional: Prize information (can be omitted if not applicable)
prizes:
  - position: 1
    value: 10000
  - position: 2
    value: 5000
  - position: 3
    value: 3000

# -----------------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------------
# Path to the competition description file (required)
description: timeseriesgym/competitions/<competition-id>/description.md

# Dataset file paths (required)
dataset:
  # Path to ground truth answers file used for grading (usually in private directory)
  answers: <competition-id>/prepared/private/answers.csv

  # Path to sample submission file provided to participants (in public directory)
  sample_submission: <competition-id>/prepared/public/sample_submission.csv

# -----------------------------------------------------------------------------------
# Grading Configuration
# -----------------------------------------------------------------------------------
grader:
  # Name of the grading metric or approach (for documentation)
  name: <grading-metric-name>

  # Path to the grading function (module:function format)
  grade_fn: timeseriesgym.competitions.<competition-id>.grade:<grading_function>

# Path to the data preparation function (module:function format)
preparer: timeseriesgym.competitions.<competition-id>.prepare:prepare
```

#### Example Configurations by Competition Type

##### 1. Standard Prediction Competition

```yaml
id: LANL-Earthquake-Prediction
name: LANL Earthquake Prediction
competition_type: simple
awards_medals: true
prizes:
  - position: 1
    value: 20000
  - position: 2
    value: 15000
  - position: 3
    value: 7000
description: timeseriesgym/competitions/LANL-Earthquake-Prediction/description.md
dataset:
  answers: LANL-Earthquake-Prediction/prepared/private/test.csv
  sample_submission: LANL-Earthquake-Prediction/prepared/public/sample_submission.csv
grader:
  name: mean-absolute-error
  grade_fn: timeseriesgym.competitions.LANL-Earthquake-Prediction.grade:grade
preparer: timeseriesgym.competitions.LANL-Earthquake-Prediction.prepare:prepare
submission_format: csv
grading_type: standard
score_higher_is_better: false
```

## Step 3: Register the Competition

Add your competition to the appropriate split files:

```bash
echo "<competition-id>" >> experiments/splits/all.txt
```

## Step 4: Test Your Implementation

### 4.1 Test Data Preparation

Test if your preparation script works correctly:

```bash
pytest tests/unit/test_data.py::test_preparing_competition_dataset_creates_non_empty_public_and_private_directories -k "<competition-id>"
```

This test verifies that your competition can be properly prepared and that both public and private directories are non-empty.

### 4.2 Test Grading Function

Add your competition to the expected scores dictionary in `tests/constants.py`:

```python
EXPECTED_SAMPLE_SUBMISSION_SCORES = {
    # Other competitions...
    "<competition-id>": <expected-score>,
}
```

The `<expected-score>` should be the score you expect from the sample submission.

Then test the grading function:

```bash
pytest tests/integration/test_eval.py::test_sample_submission_for_competition_achieves_expected_score -k "<competition-id>"
```

You might need to comment out the pytest marks to run this test:

```python
# @pytest.mark.slow
# @pytest.mark.skipif(in_ci(), reason="Avoid slow-running tests in CI.")
```

### 4.3 Manual Testing with Custom Path

If you need to specify a custom data directory, you can use this Python snippet:

```python
from pathlib import Path
from pytest import approx

from timeseriesgym.grade import grade_sample
from timeseriesgym.registry import registry

competition_id = "<competition-id>"  # Replace with your actual competition ID
data_dir = Path("/path/to/your/data/directory")
public = data_dir / competition_id / "prepared/public"

new_registry = registry.set_data_dir(data_dir)
competition = new_registry.get_competition(competition_id)

expected_score = <expected-score>  # Calculate the expected score for the sample_submission.csv

report = grade_sample(path_to_submission=public / "sample_submission.csv", competition=competition)
actual_score = report.score

assert actual_score == approx(expected_score), f"Expected {expected_score}, got {actual_score}"
```

## Step 5: Verify with Agent Run

Finally, test that your competition works with an agent:

```bash
python run_agent.py --agent-id aide --competition-set experiments/splits/custom.txt
```

Where `custom.txt` is a file containing only your competition ID.

## Troubleshooting Common Issues

### Missing or Invalid Data

- If `timeseriesgym prepare` gives errors about missing files, double-check your `prepare.py` function
- Ensure all paths are correctly specified using the `Path` object

### Grading Issues

- If the sample submission score doesn't match expectations, check if your implementation of the metric matches Kaggle's description
- For some metrics, remember to handle sign conversion (where lower is better vs. higher is better)

### Directory Structure Problems

- Make sure the competition ID is used consistently in all files
- Double-check that all required files are created in their correct locations

## Best Practices

1. **Study similar competitions** in TimeSeriesGym before adding your own
2. **Document special requirements** in your `description.md`
3. **Use relative paths** in your preparation script
4. **Keep preprocessing steps deterministic** to ensure reproducibility
5. **Add comments** explaining non-obvious data transformations

## Full Directory Structure Reference

After completion, your competition directory should look like:

```
timeseriesgym/competitions/<competition-id>/
├── config.yaml
├── description.md
├── grade.py
├── leaderboard.csv
└── prepare.py
```

And your data directory (after preparation) will look like:

```
<data-dir>/<competition-id>/
├── prepared/
│   ├── public/
│   │   ├── train.csv
│   │   ├── sample_submission.csv
│   │   └── [other training files]
│   └── private/
│       ├── test.csv
│       ├── answers.csv
│       └── [other testing files]
└── raw/
    └── [original downloaded files]
```
