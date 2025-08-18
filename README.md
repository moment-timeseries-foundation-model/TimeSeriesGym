# TimeSeriesGym: A Scalable Benchmark for Time Series Machine Learning Engineering Agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TimeSeriesGym is a comprehensive benchmarking framework for evaluating AI agents on time series machine learning engineering challenges. The current version features **34 challenges** derived from **23 unique data sources** across **8 distinct time series problems**, spanning more than **15 domains**.

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
cd timeseriesgym

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

### System Requirements

- **Python**: 3.9 or higher
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
```

Then run:

```bash
timeseriesgym grade --submission submissions.jsonl --output-dir results/
```

#### Evaluating a Single Submission

```bash
timeseriesgym grade-sample predictions.csv amp-parkinsons-disease-progression-prediction
```

#### Evaluating Hyperparameter Search Results

```bash
timeseriesgym grade-hyperparameter-search search_results/ optiver-realized-volatility-prediction-hyperparameter-search
```

#### Evaluating Code Submissions

```bash
timeseriesgym grade-code solution.py mit-bih-arrhythmia
```

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
amp-parkinsons-disease-progression-prediction
context-is-key-moirai
g2net-gravitational-wave-detection
optiver-realized-volatility-prediction-hyperparameter-search
ptb-xl-classification-challenge-feature-enhancement
stomp-R-to-python
```

### Full Competition Set

The complete set of 34 challenges is available in `TimeSeriesGym/experiments/splits/all.txt`.

## Docker Environment

TimeSeriesGym provides a Docker environment for reproducible agent execution and evaluation:

```bash
# Build the base environment
docker build --platform=linux/amd64 -t timeseriesgym-env -f environment/Dockerfile .

# Run the environment
docker run timeseriesgym-env
```

### Environment Features

- Conda environment with essential dependencies
- Pre-installed common ML packages (configurable)
- Grading server for submission validation
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
