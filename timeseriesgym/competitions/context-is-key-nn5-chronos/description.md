# Overview

## Description
In this challenge, you will evaluate the time-series foundation model [Chronos](https://arxiv.org/pdf/2403.07815) on the [Context Is Key (CiK)](https://arxiv.org/pdf/2410.18959) benchmark using a selected set of tasks. We are focusing on the [NN5](https://data.niaid.nih.gov/resources?id=zenodo_3889739) dataset, which contains daily ATM cash withdrawal data in the UK. You will:

1. Clone the CiK repository
2. Write an evaluation script for the NN5 subset of CiK
3. Submit your results

## Task

1. **Clone the CiK repository**
   - The CiK repository has been cloned into your data folder.
   ```bash
   cd context-is-key-forecasting
   pip install -r requirements.txt
   ```

2. **Every subset in CiK provides an evaluation Task instance**
   - In `cik_benchmark/tasks/nn5_tasks.py`, the `__TASKS__` list contains all tasks for the NN5 dataset. Please run the evaluation on these Task instances.

3. **Use Chronos**
   - The `experiment_chronos` function in `run_baselines.py` demonstrates how to evaluate the full CiK benchmark. Refer to its implementation to understand the repository structure.

4. **Run the evaluation on Chronous-small and save result**
   ```python
   import pickle
   from pathlib import Path

   submission_folder = Path("<submission_folder_path>")  # e.g. "./submission"

   with open(submission_folder / "submission.pkl", "wb") as f:
       pickle.dump(result, f) #result is your result file
   ```

## Submission Guidelines
- Your only required deliverable is `submission/submission.pkl`.
- The format of your result should be the same as default result format in CiK, of which you need to figure out yourself.
- Your submission should be a dictionary that maps each task name to a list of results from multiple runs. By default, each task is executed five times—-include all five runs in your output.

```python
result = {
    "Task1": [
        {"seed": 1, "score": 0},
        {"seed": 2, "score": 0},
        {"seed": 3, "score": 0},
        {"seed": 4, "score": 0},
        {"seed": 5, "score": 0}
    ],
    # ...
}
```

- You can load the provided `sample_submission.pkl` to inspect the expected structure of the output dictionary—especially useful if you’re writing your own evaluation code instead of using the CiK repository’s built-in scripts.

## Evaluation
- We will compare your `submission.pkl` against our reference output.
- **Score = 100** if your results match exactly.
- **Score = 0** if the file is missing, incorrect, or fabricated without a real run.

> ⚠ Any attempt to “hard-code” or copy the reference results without running the evaluation will be flagged as an integrity violation.
