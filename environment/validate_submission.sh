#!/bin/bash

# The agent's format check: compares a submission with the challenge's sample submission in
# /home/data. It has no access to the test answers and gives no score.
# Usage: validate_submission.sh [submission file or folder]   (default: /home/submission)
# Prints "Submission is valid. ..." (exit code 0) or "Submission invalid! <reason>" (exit code 1).
# Also installed as the command `validate_submission`.

exec /opt/conda/bin/python -I /opt/timeseriesgym/validate_submission.py "$@"
