# TimeSeriesGym: A Scalable Benchmark for Time Series Machine Learning Engineering Agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TimeSeriesGym is a comprehensive benchmarking framework for evaluating AI agents on time series machine learning engineering challenges. The current version features **33 challenges** derived from **23 unique data sources** across **8 distinct time series problems**, spanning more than **15 domains**.

Paper: https://arxiv.org/pdf/2505.13291

Beyond standard model development tasks, TimeSeriesGym evaluates AI agents on realistic ML engineering skills including:
- Data preprocessing and labeling
- Model selection and hyperparameter tuning
- Research code utilization and improvement
- Code migration between frameworks
- Feature engineering and enhancement

## Quick Start

```bash
# Install TimeSeriesGym
pip install -e .

# Prepare a lightweight set of competitions
timeseriesgym prepare --lite

# Run a sample competition
timeseriesgym grade-sample path/to/submission.csv amp-parkinsons-disease-progression-prediction
```

## Installation and Setup

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/timeseriesgym.git
cd TimeSeriesGym

# Install the package
pip install -e .
```

### Development Setup

For contributing to TimeSeriesGym, set up the development environment:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running the CI Checks Locally

GitHub Actions (`.github/workflows/ci.yml`) lints the code and runs the tests on Python 3.11 for
pushes to `main` and for pull requests. The same checks, locally:

```bash
# Lint and format check, with the ruff version CI uses (the one the dev extra pins)
pip install ruff==0.6.9
ruff check
ruff format --check

# Tests: CPU-only PyTorch first, so that pip skips the CUDA wheels
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
DEEPEVAL_TELEMETRY_OPT_OUT=YES python -m pytest -m "not network"
```

Apart from installing the dependencies, these tests need no network access and no Kaggle
credentials. The tests marked `network`, in `tests/integration/`, download and prepare real
competition data; with your Kaggle API token set up, run them with
`python -m pytest tests/integration`.

### System Requirements

- **Python**: 3.11 or higher (3.11.4 or later to run agents with `run_agent.py`)
- **Storage**: Several GB (5-20GB depending on competitions)
- **Memory**: At least 8GB RAM recommended
- **Internet**: Required for dataset downloads
- **Dependencies**: Core scientific Python libraries (NumPy, Pandas, SciPy, etc.)

## CLI Commands Overview

TimeSeriesGym provides a comprehensive command-line interface:

| Command | Description |
|---------|-------------|
| `prepare` | Download and prepare competition datasets |
| `grade` | Evaluate multiple competition submissions |
| `grade-sample` | Grade a single competition submission |
| `grade-hyperparameter-search` | Evaluate hyperparameter optimization results |
| `grade-code` | Grade a Python code submission |
| `grade-feature-enhancement` | Grade a feature-enhancement submission folder |
| `grade-predictor` | Grade a predictor folder by running it on the hidden test windows |
| `judge` | Run LLM-as-a-judge evaluation on artifacts |
| `evaluate-reasonableness` | Check if code makes a genuine modeling attempt |
| `dev` | Tools for developers extending the benchmark |
| `cleanup` | Manage disk space by removing competition files |

## Working with Competitions

### Preparing Competitions

```bash
# Prepare a single competition
timeseriesgym prepare -c amp-parkinsons-disease-progression-prediction

# Prepare all competitions (requires significant disk space)
timeseriesgym prepare -a

# Prepare TimeSeriesGym-Lite (recommended starter set)
timeseriesgym prepare --lite

# Prepare custom list of competitions
timeseriesgym prepare -l my_competitions.txt
```

**Note**: Before preparing Kaggle challenges, make sure to review the competition rules, and accept them. You will not be able to proceed without accepting the competition rules. TimeSeriesGym uses a specific version of the Kaggle API (kaggle==1.6.17). We currently do not support newer versions.

#### Additional Files

For non-Kaggle competitions, the datasets are stored using Git LFS. After installing Git LFS, run the following commands to retrieve the data stored in `timeseriesgym/data/`:

```bash
git lfs fetch --all
git lfs pull
```

For competitions where licensing restrictions prevent us from directly distributing the dataset, a script named **download\_data.py** is provided in the corresponding competition folder to help you download it manually.

#### Additional Preparation Options

- `--keep-raw`: Retain original download files (useful for debugging)
- `--data-dir=PATH`: Use custom data directory instead of default cache
- `--overwrite-checksums`: Developer option to update checksums
- `--skip-verification`: Skip checksum verification (not recommended)
- `--skip-leaderboard`: Skip leaderboard download and verification. Please add this when preparing non-kaggle competitions.

### Competition Evaluation

#### Evaluating Multiple Submissions

Create a JSONL file with submission entries:

```json
{"competition_id": "amp-parkinsons-disease-progression-prediction", "submission_path": "predictions1.csv"}
{"competition_id": "ashrae-energy-prediction", "submission_path": "predictions2.csv"}
{"competition_id": "optiver-realized-volatility-prediction-hyperparameter-search", "submission_path": "run/submission", "solution_file_path": "run/code"}
{"competition_id": "ptb-xl-classification-challenge-feature-enhancement", "submission_path": "run/submission", "solution_files_path": "run/code"}
```

For hyperparameter-search and feature-enhancement challenges, `submission_path` is the submission
folder. Hyperparameter-search entries also need `solution_file_path`: the solution code, a script
or a directory of `.py` files, which the graders require and check for a hyperparameter search.
Feature-enhancement entries may give `solution_files_path` (note the plural): the folder with
`model.py`, `train.py`, `conf/config.yaml` and the TensorBoard logs, by default the submission
folder.

`experiments/make_submission.py` writes these entries for a run group, one per run in its
`metadata.json`, failed runs included: a run without a submission is graded as invalid, so every
run counts in the report. It uses the run's `code/` directory (the agent's `/home/code`) as
`solution_file_path`, and as `solution_files_path` when the submission folder has no `model.py`.

Then run:

```bash
timeseriesgym grade --submission submissions.jsonl --output-dir results/
```

#### Evaluating a Single Submission

```bash
timeseriesgym grade-sample predictions.csv amp-parkinsons-disease-progression-prediction
```

#### Medals

Reports give Kaggle's medals (`gold_medal`, `silver_medal`, `bronze_medal`, `above_median`) and
their thresholds, ranked by Kaggle's rules against the challenge's `leaderboard.csv`; challenges
without a real leaderboard award none. The scores are TimeSeriesGym's own: most Kaggle-derived
challenges hold out their test set from the competition's training data, while the leaderboard was
scored on Kaggle's test set, so a medal places a score only approximately. Three challenges differ
more, and their descriptions say so:

- **LANL-Earthquake-Prediction**: the test is 1,112 segments of 150,000 rows, as long as Kaggle's,
  but cut from the last 5 earthquake cycles of Kaggle's training data; the leaderboard was scored on
  Kaggle's own 2,624 test segments.
- **amp-parkinsons-disease-progression-prediction**: the test set is 10% of the patients in
  Kaggle's training data; the leaderboard was scored on Kaggle's hidden test set.
- **ashrae-energy-prediction**: the test set is the last two months of 2016; the leaderboard was
  scored on Kaggle's test data from 2017 and 2018.

#### Evaluating Hyperparameter Search Results

```bash
timeseriesgym grade-hyperparameter-search run/submission/ \
  optiver-realized-volatility-prediction-hyperparameter-search \
  --solution-file-path run/code/
```

`--solution-file-path` is required: without the solution code (a script or a directory of `.py`
files) the submission is graded as invalid.

#### Evaluating Feature-Enhancement Submissions

```bash
timeseriesgym grade-feature-enhancement run/submission/ \
  ptb-xl-classification-challenge-feature-enhancement \
  --solution-files-path run/code/
```

The score is a dict of sub-scores (TensorBoard, code quality, Hydra configuration, accuracy) and
their mean, `overall_score`. Without `--solution-files-path`, the code, configuration and logs are
read from the submission folder.

#### Evaluating Code Submissions

```bash
timeseriesgym grade-code solution.py mit-bih-arrhythmia
```

#### Evaluating Predictor Submissions

In gift-eval-short-horizon and time-series-library-long-horizon-forecast the test labels are not
in the public data: the agent submits a predictor, the folder `submission/predictor/` with
`predict.py`, and the grader runs it on the hidden test windows (see `timeseriesgym/predictor`).

```bash
timeseriesgym grade-predictor run/submission/predictor gift-eval-short-horizon
```

The predictor runs in the grading sandbox (below), with the Python interpreter named by
`--python` or `$TSGYM_PREDICTOR_PYTHON`, else the grader's own. Use the agent environment's
interpreter, so that the predictor meets the packages it was written with: the grader image
sets `$TSGYM_PREDICTOR_PYTHON` to its copy of the agents' conda environment
(see [Docker Environment](#docker-environment)). `--trusted` runs the predictor as a plain
subprocess instead, which can read whatever the grading user can, the answers included: for
trusted predictors only.

#### Agent Code Runs in a Sandbox

Code challenges import and run the submitted `submission.py`, hyperparameter-search challenges
load the submitted models (unpickling and `torch.jit.load` can run code), and predictor
challenges run the submitted predictor. The graders do this in a separate, confined process,
never in the grading process, which holds the answers.
That process gets a copy of the submission and of the public input data, no credentials, no
sockets, and limits on time, memory and file size, and it cannot read the challenge's private
files. The grading process then compares what it computed with the answers. A submission whose
code crashes, hangs or exceeds a limit is invalid.

- Graded as root (e.g. in a container), agent code runs as the user `sandbox`. Create it with
  `useradd --system --no-create-home sandbox`, or name another user in
  `TIMESERIESGYM_SANDBOX_USER`. Grading as root without such a user stops with an error. The
  grader image has this user, and `run_agent.py` grades in it as root.
- Graded as a normal user, agent code runs as that user, and Landlock (Linux 5.13 or later)
  keeps it from the answers and the user's files. This is a narrower boundary than a container
  (the sandbox shares `/dev/shm` and the process quota with the user's programs), so grade
  untrusted submissions as root in a container. Where Landlock is not available, grading stops
  with an error, as agent code could read the answers; `TIMESERIESGYM_SANDBOX_UNCONFINED=1`
  grades trusted submissions anyway.
- `TIMESERIESGYM_SANDBOX_TIMEOUT_SEC` sets the time limit (default: 1800 s for code challenges,
  600 s for hyperparameter searches; a predictor's comes from its challenge's
  `predictor_config`), and `TIMESERIESGYM_SANDBOX_MEMORY_MB` the memory limit for all of the
  sandbox's processes together (default: 3/4 of the machine's or container's memory).

`timeseriesgym/sandbox.py` describes what each mode does and does not protect against.

#### Evaluating Code Reasonableness

The `evaluate-reasonableness` command uses an LLM-as-a-judge and simple rule-based heuristics to determine whether code makes a genuine modeling attempt rather than gaming the submission format (e.g., copying sample submissions, using random predictions).

```bash
# Basic usage
timeseriesgym evaluate-reasonableness path/to/code/folder

# With custom model
timeseriesgym evaluate-reasonableness path/to/code/folder --model gpt-4o

# JSON output
timeseriesgym evaluate-reasonableness path/to/code/folder --json

# Verbose output
timeseriesgym evaluate-reasonableness path/to/code/folder --verbose
```

**Requirements**: Set `OPENAI_API_KEY` environment variable.

**Output includes**:
- `is_reasonable`: Whether the code is a genuine modeling attempt
- `score`: Evaluation score
- `threshold`: Dynamic threshold based on code patterns
- `has_suspicious_patterns`: Regex detection of non-genuine patterns
- `has_modeling_patterns`: Regex detection of genuine ML code

### Managing Disk Space

```bash
# Remove zip files only
timeseriesgym cleanup -c competition-id

# Complete cleanup of all competitions (with confirmation)
timeseriesgym cleanup -a --full
```

## Competition Sets

### TimeSeriesGym-Lite

A carefully curated subset of six challenges designed for efficient evaluation while maintaining coverage across domains and problem types:

```
# Available in TimeSeriesGym/experiments/splits/lite.txt
child-mind-institute-detect-sleep-states
MIT-BIH-Arrhythmia-Weak-Supervision
moment-anomaly-detection-score
ptb-xl-classification-challenge-hyperparameter-search
optiver-realized-volatility-prediction-missing
resnet-tensorflow-to-pytorch
```

### Full Competition Set

The complete set of 33 gradeable challenges is available in
`TimeSeriesGym/experiments/splits/all.txt`.

## Docker Environment

TimeSeriesGym provides two Docker images for reproducible agent execution and evaluation, both
built from `environment/Dockerfile`: the agents' base environment and the grader.

```bash
# Build the base environment of the agents' images (the default target)
docker build --platform=linux/amd64 -t timeseriesgym-env -f environment/Dockerfile .

# Build the grader
docker build --platform=linux/amd64 --target grader -t timeseriesgym-grader \
  -f environment/Dockerfile .
```

An agent's container gets the challenge's public data, at `/home/data`, and nothing else from
TimeSeriesGym: no private data, no graders and no TimeSeriesGym code. After the agent's container
has stopped, `run_agent.py` grades the run in a separate container of `timeseriesgym-grader`, the
only one that gets the challenge's private data (read-only, without network). See
[agents/README.md](agents/README.md#grading-workflow), and its migration notes if you built agent
images before this change.

The grader image has the user `sandbox`, which runs agent code while grading (see
[Agent Code Runs in a Sandbox](#agent-code-runs-in-a-sandbox)), and a copy of the agents' conda
environment `agent` at `/opt/conda/envs/agent`, with which it runs predictors
(`$TSGYM_PREDICTOR_PYTHON`). The copy comes from the `env` stage of the same Dockerfile, so it
has the packages of `environment/requirements.txt`; to grade with an agent image's environment
instead (packages its own Dockerfile adds), build the grader from that image:

```bash
docker build --platform=linux/amd64 --target grader --build-arg PREDICTOR_ENV=aide \
  -t timeseriesgym-grader-aide -f environment/Dockerfile .
```

and pass `--grader-image timeseriesgym-grader-aide` to `run_agent.py`. Packages an agent installs
while it runs are in neither: give them to its image.

The agent's image copies only files of `environment/`. The grader copies the build context, which
`.dockerignore` limits to the `timeseriesgym` package, `environment/`, `pyproject.toml`,
`README.md` and `LICENSE`, without `timeseriesgym/data/` (the Git LFS archive of raw challenge
data, which holds test answers). Everything else in the checkout, such as `runs/`, prepared data
and `.env` files, stays out of both images. Do not rely on other repository files inside them.

### Environment Features

- Conda environment with essential dependencies
- Pre-installed common ML packages (configurable)
- `validate_submission`: a format check against the sample submission (no answers, no score)
- Agent instruction templates

To build without heavy dependencies:

```bash
docker build --platform=linux/amd64 -t timeseriesgym-env -f environment/Dockerfile --build-arg INSTALL_HEAVY_DEPENDENCIES=false .
```

## Agent Integration

TimeSeriesGym supports multiple agent scaffolds:

- **AIDE**: AI Developer Environment
- **ResearchAgent**: Specialized for research tasks
- **OpenHands**: General-purpose agent framework

Detailed information on agent evaluation is available in [agents/README.md](agents/README.md).

Agent containers have no network and no API keys: agents call their models through a model proxy
that runs on the host for each run and holds the keys (see
[Network and Model Access](agents/README.md#network-and-model-access)).

## Extending TimeSeriesGym

Documentation for adding new challenges is available in the [documentation/](documentation) directory.

## Experiments

The `experiments/` directory contains resources from our publication:

- Competition splits in `experiments/splits/`
- Submission compilation script in `experiments/make_submission.py`

## Acknowledgements
We would like to thank the authors of [MLE-Bench](https://github.com/openai/mle-bench) for providing an excellent code repository that we could build on. Their thoughtful design choices, and open-source approach have been instrumental in enabling TimeSeriesGym.

We also acknowledge the organizers of various competition and dataset providers whose work has been incorporated into TimeSeriesGym. Their commitment to advancing machine learning through public benchmarks has made this project possible.

## Citation

If you use TimeSeriesGym in your research, please cite:

```bibtex
@article{cai2025timeseriesgym,
  title={TimeSeriesGym: A Scalable Benchmark for(Time Series) Machine Learning Engineering Agents},
  author={Cai, Yifu and Li, Xinyu and Goswami, Mononito and Wili{\'n}ski, Micha{\l} and Welter, Gus and Dubrawski, Artur},
  year={2025},
  primaryClass={cs.CL},
}
```

## License

TimeSeriesGym is released under the MIT License.
