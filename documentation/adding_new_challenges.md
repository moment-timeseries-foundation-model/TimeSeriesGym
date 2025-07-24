# Adding a New Non-Kaggle Competition to TimeSeriesGym

This guide focuses on adding custom or non-Kaggle competitions to TimeSeriesGym. While the process is similar to adding Kaggle competitions, there are important differences in data acquisition, preparation, and validation.

## Overview

TimeSeriesGym supports competitions from various sources beyond Kaggle, including:
- Academic datasets (e.g., UCR/UEA time series archives)
- Research repositories (e.g., PhysioNet)
- Custom time series challenges
- Converted competitions from other platforms

## Step 1: Create Competition Directory

Create a directory for your competition:

```bash
mkdir -p timeseriesgym/competitions/<competition-id>
```

> **Important**: Choose a clear, descriptive `competition-id` that follows the naming convention of other competitions (kebab-case recommended).

Add your competition to the non-Kaggle competitions list in `constants.py`:

## Step 2: Data Acquisition

Unlike Kaggle competitions, non-Kaggle data requires custom acquisition. There are two approaches:

### Option A: Programmatic Download (Recommended)

Create a `download_data.py` script that programmatically downloads and compresses the data:

```python
import os
import requests
import zipfile
from pathlib import Path

def download_dataset(output_dir: Path) -> Path:
    """
    Download dataset files and create a zip archive.

    Args:
        output_dir: Directory where the zip file will be saved

    Returns:
        Path to the created zip file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "<competition-id>.zip"

    # Create a temporary directory for downloaded files
    temp_dir = output_dir / "temp_download"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Download all required files
        files_to_download = [
            ("https://example.com/dataset/train.csv", "train.csv"),
            ("https://example.com/dataset/test.csv", "test.csv"),
            # Add all files needed
        ]

        for url, filename in files_to_download:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            file_path = temp_dir / filename
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded {filename}")

        # Create a zip file containing all downloaded files
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in temp_dir.glob('*'):
                zipf.write(file_path, arcname=file_path.name)

        print(f"Created zip archive at {zip_path}")
        return zip_path

    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    download_dataset(Path("./"))
```

For examples, refer to existing implementations:
- `MIT-BIH-Arrhythmia/download_data.py`
- `ucr-anomaly-detection/download_data.py`

### Option B: Manual Download

If programmatic download is not feasible:

1. Download the dataset files manually
2. Create a zip archive containing all necessary files
3. Place the zip file at `timeseriesgym/competitions/<competition-id>/<competition-id>.zip`

> **Important**: When downloading data manually, be consistent with file naming and organization to ensure reproducibility.

### Checksum Considerations

TimeSeriesGym uses checksums to verify data integrity. Be aware that:

- Different download methods (curl, wget, browsers) may produce files with different checksums
- Ensure the download method produces consistent results
- You can use `--overwrite-checksums` during first preparation to generate new checksums

## Step 3: Create Required Files

### 3.1 Competition Description (`description.md`)

Create a detailed `description.md` file that clearly explains:

- Competition overview and objective
- Dataset description and organization
- Submission format requirements
- Evaluation metrics
- Any special instructions or requirements

Since non-Kaggle competitions may not have standardized descriptions, use this template:

```markdown
# Competition Name

## Overview
[Brief introduction to the problem]

## Dataset Description
[Explain the data, including file formats, fields, and their meanings]

## Submission Format
[Clearly specify the expected format for submissions]

## Evaluation
[Explain the evaluation metric in detail]

## Data Source
[Provide attribution and links to the original data source]
```

### 3.2 Data Preparation Script (`prepare.py`)

Create a `prepare.py` file that extracts and processes the data:

```python
import shutil
from pathlib import Path
import pandas as pd
import zipfile
# Additional imports as needed

def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Process the raw competition files into public (training) and private (testing) directories.

    Args:
        raw: Path to the directory containing raw downloaded files
        public: Path to the directory where public (training) files should be placed
        private: Path to the directory where private (testing) files should be placed
    """
    # Extract data if it's still in zip format
    zip_files = list(raw.glob("*.zip"))
    if zip_files:
        with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
            zip_ref.extractall(raw)

    # Process and organize data
    # 1. Copy/transform training data to public directory
    # 2. Copy/transform test data to private directory
    # 3. Generate any additional necessary files

    # Example: Train/test split if data is not already split
    # data = pd.read_csv(raw / "full_dataset.csv")
    # train_data = data.sample(frac=0.8, random_state=42)
    # test_data = data.drop(train_data.index)
    # train_data.to_csv(public / "train.csv", index=False)
    # test_data.to_csv(private / "test.csv", index=False)

    # Create a sample submission file if not present
    if not (raw / "sample_submission.csv").exists():
        # Create a sample submission based on test data
        test_data = pd.read_csv(private / "test.csv")
        sample_sub = pd.DataFrame()
        sample_sub['id'] = test_data['id']
        sample_sub['target'] = 0  # Default values
        sample_sub.to_csv(public / "sample_submission.csv", index=False)
    else:
        shutil.copy(raw / "sample_submission.csv", public / "sample_submission.csv")

    # Create the answers file for evaluation
    # answers = pd.DataFrame()
    # answers['id'] = test_data['id']
    # answers['target'] = test_data['target']
    # answers.to_csv(private / "answers.csv", index=False)
```

### 3.3 Competition-Specific Grading

TimeSeriesGym supports different types of grading based on what aspect of ML engineering is being evaluated. Your competition might focus on one of these areas:

#### Standard Prediction Grading (`grade.py`)

For competitions evaluating prediction quality:

```python
import pandas as pd
import numpy as np
# Additional imports as needed

def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Evaluate a prediction submission against ground truth.

    Args:
        submission: DataFrame containing the predictions
        answers: DataFrame containing the ground truth

    Returns:
        float: Score value (higher is better, unless specified otherwise in config.yaml)
    """
    # Implement the evaluation metric
    # Example metrics:

    # For classification:
    # from sklearn.metrics import accuracy_score
    # return accuracy_score(answers['target'], submission['prediction'])
```

#### Hyperparameter Search Grading

For competitions evaluating hyperparameter optimization:

```python
import pandas as pd
import json
from pathlib import Path

def grade_hyperparameter_search(folder_path: Path, answers_file: Path) -> float:
    """
    Evaluate a hyperparameter search by examining multiple submissions.

    Args:
        folder_path: Path to folder containing multiple submission files
        answers_file: Path to the answers file

    Returns:
        float: Score representing the quality of hyperparameter optimization
    """
    # Load the answers for reference
    answers = pd.read_csv(answers_file)

    # Find all submission files in the folder
    submission_files = list(folder_path.glob("*.csv"))

    # Track scores across all submissions
    scores = []
    for submission_file in submission_files:
        submission = pd.read_csv(submission_file)
        # Calculate individual score for this submission
        score = calculate_score(submission, answers)
        scores.append(score)

    # Metrics could include:
    # - Best score achieved
    # - Convergence rate
    # - Diversity of explored hyperparameters
    # - Efficiency of search

    # Return the best score achieved
    return max(scores)

def calculate_score(submission, answers):
    # Individual submission scoring logic
    pass
```

#### Code Quality Grading

For competitions evaluating code implementation:

```python
import importlib.util
import inspect
from pathlib import Path

def grade_code(code_file: Path, reference_solution: Path = None) -> float:
    """
    Evaluate code quality, correctness, and efficiency.

    Args:
        code_file: Path to the submitted code file
        reference_solution: Optional path to reference implementation

    Returns:
        float: Score representing code quality
    """
    # Load the submitted module
    spec = importlib.util.spec_from_file_location("submission", code_file)
    submission_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submission_module)

    # Evaluation metrics could include:
    # - Functional correctness (runs without errors)
    # - Output correctness (matches expected values)
    # - Time complexity (execution speed)
    # - Space complexity (memory usage)
    # - Code quality (static analysis)

    score = evaluate_correctness(submission_module)
    # Add additional evaluation criteria as needed

    return score

def evaluate_correctness(module):
    # Test module on reference inputs/outputs
    pass
```

Choose the appropriate grading approach based on your competition's focus and ensure your `config.yaml` specifies which type of grading to use.

### 3.4 Leaderboard File (Optional)

For non-Kaggle competitions, a leaderboard is optional:

- If a public leaderboard exists, create a `leaderboard.csv` with the same format as Kaggle competitions
- If no leaderboard exists, skip this file and use the `--skip-leaderboard` flag during preparation

### 3.5 Configuration File (`config.yaml`)

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

# Optional: The type of grading to use (standard, hyperparameter, code, migration, feature)
grading_type: standard

# -----------------------------------------------------------------------------------
# Optional: Competition-Specific Configurations
# -----------------------------------------------------------------------------------

# For hyperparameter search competitions
hyperparameter_search_config:
  # Name of the baseline submission file
  baseline_submission_name: baseline_submission.csv

  # Name of the improved submission file
  improved_submission_name: submission.csv

  # Name of the baseline model file
  baseline_model_name: baseline_model.pkl

  # Name of the improved model file
  improved_model_name: model.pkl

# For code implementation competitions
coding_config:
  # Name of the coding task
  name: <task-name>

  # Directory containing input data
  input_data_dir: <competition-id>/prepared/public/

  # Required functions and their specifications
  required_functions:
    <function_name>:
      # Whether arguments must exactly match the specified order
      exact_match: true

      # Required arguments and their values/sources
      required_args:
        <arg1>:
          file_name: <data_file.ext>
        <arg2>: <value>

      # Expected return type(s)
      expected_output_type: [<type1>, <type2>]

      # Expected shape of outputs (for array-like returns)
      expected_output_shape: [[<dim1>, <dim2>], [<dim3>]]

      # Expected accuracy or performance threshold
      expected_accuracy: 0.8

  # Required classes and their specifications
  required_classes:
    <class_name>:
      # Required initialization parameters
      init_args:
        <param1>: <value1>

      # Required methods
      required_methods:
        <method_name>:
          required_args:
            <arg1>: <value1>

# For feature engineering competitions
feature_enhancement_config:
  # Path to model implementation
  model_code_file: model.py

  # Path to configuration file
  config_file: config.yaml

  # Path to training script
  training_file: train.py
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
grading_type: standard
```

##### 2. Code Implementation Competition

```yaml
id: stomp-R-to-python
name: Convert STOMP Algorithm Implementation in R to Python
competition_type: simple
awards_medals: false
description: timeseriesgym/competitions/stomp-R-to-python/description.md
dataset:
  answers: stomp-R-to-python/prepared/private/test_results.csv
  sample_submission: stomp-R-to-python/prepared/public/sample_submission.py
grader:
  name: python-code-grading
  grade_fn: timeseriesgym.competitions.stomp-R-to-python.grade:grade_code
preparer: timeseriesgym.competitions.stomp-R-to-python.prepare:prepare
grading_type: code
coding_config:
  name: python-stomp
  input_data_dir: stomp-R-to-python/prepared/public/
  required_functions:
    stomp:
      exact_match: true
      required_args:
        ref_data:
          file_name: ref_data.npy
        query_data: null
        window_size: 30
        exclusion_zone: 0.5
        verbose: 2
      expected_output_type: [numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
      expected_output_shape: [[521, ], [521, ], [521, ], [521, ], [521, ], [521, ]]
```

##### 3. Feature Engineering Competition

```yaml
id: ptb-xl-classification-challenge-feature-enhancement
name: PTB-XL ECG Classification with Feature Enhancement
competition_type: simple
awards_medals: false
description: timeseriesgym/competitions/ptb-xl-classification-challenge-feature-enhancement/description.md
parent_id: ptb-xl-classification-challenge
dataset:
  answers: ptb-xl-classification-challenge-feature-enhancement/prepared/private/test_label_df.csv
  sample_submission: ptb-xl-classification-challenge-feature-enhancement/prepared/public/sample_submission.csv
grader:
  name: custom-feature-enhancement-grading
  grade_fn: timeseriesgym.competitions.ptb-xl-classification-challenge-feature-enhancement.grade:grade_feature_enhancement
grading_type: feature
feature_enhancement_config:
  model_code_file: model.py
  config_file: config.yaml
  training_file: train.py
```

##### 4. Hyperparameter Search Competition

```yaml
id: optiver-realized-volatility-prediction-hyperparameter-search
name: Optiver Realized Volatility Prediction with Hyperparameter Search
competition_type: simple
awards_medals: true
description: timeseriesgym/competitions/optiver-realized-volatility-prediction-hyperparameter-search/description.md
parent_id: optiver-realized-volatility-prediction
dataset:
  answers: optiver-realized-volatility-prediction-hyperparameter-search/prepared/private/test.csv
  sample_submission: optiver-realized-volatility-prediction-hyperparameter-search/prepared/public/sample_submission.csv
grader:
  name: custom-hyperparameter-search-grading
  grade_fn: timeseriesgym.competitions.optiver-realized-volatility-prediction-hyperparameter-search.grade:grade_hyperparameter_search
grading_type: hyperparameter
hyperparameter_search_config:
  baseline_submission_name: baseline_submission.csv
  improved_submission_name: submission.csv
  baseline_model_name: baseline_model.pkl
  improved_model_name: model.pkl
```

## Step 4: Register the Competition

Add your competition to the appropriate split files:

```bash
echo "<competition-id>" >> experiments/splits/all.txt
```

Consider adding it to custom splits for specific types of time series tasks:

```bash
echo "<competition-id>" >> experiments/splits/custom_splits/anomaly_detection.txt
```

## Step 5: Test Your Implementation

### 5.1 Prepare the Dataset

Prepare the dataset with the appropriate flags based on your competition type:

```bash
# For standard prediction competitions
timeseriesgym prepare -c <competition-id> --skip-leaderboard

# For code-based or hyperparameter competitions
timeseriesgym prepare -c <competition-id> --skip-leaderboard --allow-empty-private-dir --allow-empty-answers-file
```

The additional flags address common issues with non-Kaggle competitions:

- `--skip-leaderboard`: Skips leaderboard verification if none exists
- `--allow-empty-private-dir`: Allows competitions where test data is generated procedurally
- `--allow-empty-answers-file`: Allows competitions where answers are generated during preparation

### 5.2 Test the Appropriate Grading Function

Different competition types require different testing approaches:

#### For Standard Prediction Competitions

Use the standard approach to test the grading function:

```bash
pytest tests/integration/test_eval.py::test_sample_submission_for_competition_achieves_expected_score -k "<competition-id>"
```

#### For Hyperparameter Search Competitions

Create a dedicated test script:

```python
from pathlib import Path
from timeseriesgym.grade import grade_hyperparameter_search
from timeseriesgym.registry import registry

# Set up paths
competition_id = "<competition-id>"
data_dir = Path("/path/to/your/data/directory")
new_registry = registry.set_data_dir(data_dir)
competition = new_registry.get_competition(competition_id)

# Create test hyperparameter folder with sample submissions
test_folder = Path("./test_hyperparams")
test_folder.mkdir(exist_ok=True)

# Create sample submissions with different parameters
for i in range(3):
    # Create sample submissions with varying hyperparameters
    # ... (implementation details)

# Test grading function
report = grade_hyperparameter_search(test_folder, competition)
print(f"Hyperparameter search score: {report.score}")
```

#### For Code Implementation Competitions

Test the code grading function:

```python
from pathlib import Path
from timeseriesgym.grade import grade_code
from timeseriesgym.registry import registry

competition_id = "<competition-id>"
data_dir = Path("/path/to/your/data/directory")
new_registry = registry.set_data_dir(data_dir)
competition = new_registry.get_competition(competition_id)

# Test with a reference solution
report = grade_code(
    Path("/path/to/sample_solution.py"),
    competition,
    reference_solution=Path("/path/to/reference_implementation.py")
)
print(f"Code implementation score: {report.score}")
```

#### For Framework Migration Competitions

Test the migration grading function:

```python
from pathlib import Path
from timeseriesgym.grade import grade_migration
from timeseriesgym.registry import registry

competition_id = "<competition-id>"
data_dir = Path("/path/to/your/data/directory")
new_registry = registry.set_data_dir(data_dir)
competition = new_registry.get_competition(competition_id)

# Test with a sample migration
report = grade_migration(
    Path("/path/to/migrated_implementation.py"),
    competition,
    reference_code=Path("/path/to/original_implementation.py")
)
print(f"Migration quality score: {report.score}")
```

### 5.3 Run a Test Agent on the Specific Competition Type

Use an appropriate agent for your competition type:

```bash
# For standard prediction or general ML tasks
python run_agent.py --agent-id aide --competition-set experiments/splits/single.txt

# For code implementation or migration tasks
python run_agent.py --agent-id mlagentbench --competition-set experiments/splits/single.txt

# For more complex research tasks
python run_agent.py --agent-id openhands --competition-set experiments/splits/single.txt
```

Where `single.txt` contains only your competition ID.

## Troubleshooting

### Common Issues with Non-Kaggle Competitions

1. **Data consistency issues**:
   - Solution: Use deterministic data preparation steps

2. **Checksum verification failures**:
   - Solution: Use `--overwrite-checksums` on first preparation

3. **Missing leaderboard**:
   - Solution: Use `--skip-leaderboard` flag

4. **Empty directories**:
   - Solution: Use `--allow-empty-private-dir` and `--allow-empty-answers-file` flags

5. **Data format incompatibilities**:
   - Solution: Standardize formats in the preparation script

6. **Issues with grading function selection**:
   - Solution: Ensure `grading_type` in `config.yaml` matches the implemented function

7. **Agent failures on non-standard competitions**:
   - Solution: Provide clearer instructions in `description.md`
   - Solution: Include example submissions in the `public` directory

## Example: Converting a Framework Migration Task

Here's a simplified example of creating a framework migration competition (TensorFlow to PyTorch):

```python
# download_data.py
def download_dataset(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary directory
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Create or download the TensorFlow model
    tf_model_path = temp_dir / "model.h5"
    create_tensorflow_model(tf_model_path)

    # Create test cases
    create_test_cases(temp_dir)

    # Package everything in a zip
    zip_path = output_dir / "resnet-tensorflow-to-pytorch.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in temp_dir.glob('*'):
            zipf.write(file_path, arcname=file_path.name)

    return zip_path

# prepare.py
def prepare(raw, public, private):
    # Extract dataset
    with zipfile.ZipFile(list(raw.glob("*.zip"))[0], 'r') as zip_ref:
        zip_ref.extractall(raw)

    # Copy TensorFlow model and code to public
    shutil.copy(raw / "model.h5", public / "model.h5")
    shutil.copy(raw / "tensorflow_model.py", public / "tensorflow_model.py")

    # Create a template for PyTorch implementation
    with open(public / "pytorch_template.py", 'w') as f:
        f.write("""
import torch
import torch.nn as nn

class ResNetPyTorch(nn.Module):
    def __init__(self):
        super(ResNetPyTorch, self).__init__()
        # TODO: Implement PyTorch model architecture

    def forward(self, x):
        # TODO: Implement forward pass
        pass

def load_model():
    # TODO: Implement model loading
    pass

def predict(x):
    # TODO: Implement prediction function
    pass
""")

    # Copy test cases to private for grading
    shutil.copy(raw / "test_cases.npy", private / "test_cases.npy")
    shutil.copy(raw / "expected_outputs.npy", private / "expected_outputs.npy")

    # Create a sample submission
    shutil.copy(raw / "pytorch_reference.py", public / "sample_submission.py")

# grade.py
def grade_migration(migrated_code, reference_outputs_path):
    """Grade a PyTorch implementation against TensorFlow reference outputs"""
    # Import the submitted PyTorch model
    spec = importlib.util.spec_from_file_location("submission", migrated_code)
    pytorch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pytorch_module)

    # Load test cases and reference outputs
    import numpy as np
    test_cases = np.load(reference_outputs_path.parent / "test_cases.npy")
    expected_outputs = np.load(reference_outputs_path)

    # Convert test cases to PyTorch tensors
    import torch
    test_tensors = [torch.tensor(case, dtype=torch.float32) for case in test_cases]

    # Get predictions from PyTorch model
    pytorch_outputs = []
    for tensor in test_tensors:
        pytorch_output = pytorch_module.predict(tensor)
        pytorch_outputs.append(pytorch_output.detach().numpy())

    # Calculate similarity scores
    similarities = []
    for expected, actual in zip(expected_outputs, pytorch_outputs):
        similarity = 1.0 - np.mean(np.abs(expected - actual))
        similarities.append(similarity)

    # Return average similarity
    return np.mean(similarities)
```

## Final Checklist

Before submitting your competition, verify that:

- [ ] All required files are present and correctly formatted
- [ ] Data is properly split between public and private
- [ ] The competition can be prepared with appropriate flags
- [ ] The grading function correctly implements the evaluation metric
- [ ] The description clearly explains all aspects of the competition
- [ ] Sample submissions can be properly evaluated
- [ ] An agent can successfully run on the competition
- [ ] All necessary citations and attributions are included
- [ ] The `config.yaml` file correctly specifies the grading approach
- [ ] The appropriate grading function is implemented and tested appropriate flags:

```bash
timeseriesgym prepare -c <competition-id> --skip-leaderboard --allow-empty-private-dir --allow-empty-answers-file
```

The additional flags address common issues with non-Kaggle competitions:

- `--skip-leaderboard`: Skips leaderboard verification if none exists
- `--allow-empty-private-dir`: Allows competitions where test data is generated procedurally
- `--allow-empty-answers-file`: Allows competitions where answers are generated during preparation

### 5.2 Test the Grading Function

Follow the testing approach described in the Kaggle competition guide, but be aware of these differences:

- You may need to create custom test cases for unique evaluation metrics
- For competitions without standard benchmarks, establish baseline performance
- Document expected performance ranges in the competition description

### 5.3 Run a Test Agent

Run a test agent on your competition:

```bash
python run_agent.py --agent-id aide --competition-set experiments/splits/single.txt
```

Where `single.txt` contains only your competition ID.

## Special Considerations for Non-Kaggle Competitions

### Time Series-Specific Issues

- **Variable-length sequences**: Ensure proper padding/truncation is documented
- **Multivariate data**: Clearly specify dimensions and formats
- **Different sampling rates**: Document any resampling requirements
- **Missing values**: Specify how they should be handled

### Custom Metrics

For specialized time series metrics:

1. Implement the metric in `grade.py`
2. Include detailed explanation in `description.md`
3. Provide a reference implementation if possible
4. Document edge cases (ties, invalid submissions, etc.)

### Data Licensing

Always include proper attribution and licensing information:

1. Document the original data source
2. Include any required citations
3. Note any usage restrictions
4. Verify you have permission to redistribute the data

## Troubleshooting

### Common Issues with Non-Kaggle Competitions

1. **Data consistency issues**:
   - Solution: Use deterministic data preparation steps

2. **Checksum verification failures**:
   - Solution: Use `--overwrite-checksums` on first preparation

3. **Missing leaderboard**:
   - Solution: Use `--skip-leaderboard` flag

4. **Empty directories**:
   - Solution: Use `--allow-empty-private-dir` and `--allow-empty-answers-file` flags

5. **Data format incompatibilities**:
   - Solution: Standardize formats in the preparation script

## Final Checklist

Before submitting your competition, verify that:

- [ ] All required files are present and correctly formatted
- [ ] Data is properly split between public and private
- [ ] The competition can be prepared with appropriate flags
- [ ] The grading function correctly implements the evaluation metric
- [ ] The description clearly explains all aspects of the competition
- [ ] Sample submissions can be properly evaluated
- [ ] An agent can successfully run on the competition
- [ ] All necessary citations and attributions are included
