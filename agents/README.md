# Agent Integration Guide

This guide details how to run and evaluate various AI agents with TimeSeriesGym. We provide integrations with leading open-source agents and a diagnostic dummy agent for environment verification.

## Supported Agents

| Agent | ID | Description | Source |
|-------|----|----|--------|
| Dummy | `dummy` | Diagnostic agent for environment testing | N/A |
| [AIDE](https://www.weco.ai/blog/technical-report) | `aide` | AI Developer Environment | [GitHub Fork](https://github.com/moment-timeseries-foundation-model/aideml) |
| [OpenHands](https://arxiv.org/abs/2407.16741) | `openhands` | General-purpose agent framework | [OpenHands v0.34.0](https://github.com/All-Hands-AI/OpenHands/tree/0.34.0) |
| [MLAgentBench](https://openreview.net/forum?id=1Fs1LvjYQW) | `mlagentbench` | Research-oriented agent framework | [GitHub Fork](https://github.com/JunShern/MLAgentBench) |

## Setup Requirements

### System Prerequisites

- [Docker](https://docs.docker.com/engine/install/) - Required for all agents
- [Sysbox](https://github.com/nestybox/sysbox) - Secure container runtime (strongly recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) - Required for GPU acceleration

### Building Agent Images

First build the agents' base environment and the grader, from the repository root:

```bash
docker build --platform=linux/amd64 -t timeseriesgym-env -f environment/Dockerfile .
docker build --platform=linux/amd64 --target grader -t timeseriesgym-grader \
  -f environment/Dockerfile .
```

Set environment variables for the container directories:

```bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent
```

Build an agent image:

```bash
docker build --platform=linux/amd64 -t <agent-id> agents/<agent-id>/ \
  --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
  --build-arg LOGS_DIR=$LOGS_DIR \
  --build-arg CODE_DIR=$CODE_DIR \
  --build-arg AGENT_DIR=$AGENT_DIR
```

Replace `<agent-id>` with one of: `dummy`, `aide`, `mlagentbench`, or `openhands`.

## Running Agents

TimeSeriesGym provides the `run_agent.py` script to execute agents on competition sets:

```bash
python run_agent.py --agent-id <agent-id> --competition-set <competition-set-file> [OPTIONS]
```

### Competition Sets

The `experiments/splits/` directory contains predefined competition sets:

- `all.txt` - Complete set of all competitions
- `lite.txt` - Curated lightweight competition set
- `prepare_lit.txt` - Some competitions in the lite split are derived from parent competitions. This split includes all the competitions which must be prepare in order to run experiments on the lite datasets.
- `dev.txt` - Small set for development testing


### Example Execution

Run AIDE on a development competition set:

```bash
python run_agent.py --agent-id aide --competition-set experiments/splits/dev.txt
```

### Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Custom data directory | `.cache` |
| `--container-config` | Custom container configuration | `environment/config/container_configs/default.json` |
| `--allow-empty-private-dir` | Allow competitions with empty private directories | `False` |
| `--allow-empty-answers-file` | Allow competitions with empty answers files | `False` |
| `--grader-image` | Image of the grading step | `timeseriesgym-grader` |
| `--skip-grading` | Do not grade the runs (grade them with the manual workflow below) | `False` |
| `--grading-timeout` | Seconds the grading of one run may take | `7200` |

### Container Configuration

The default container configuration is located at `environment/config/container_configs/default.json`. You can customize:

- Resource limits (CPU, memory)
- Mounted volumes
- Environment variables
- GPU passthrough

To use GPUs, add `"gpus": "device_ids"` to your container config. For example:

```json
{
  "gpus": "0,1",
  "other_config": "values"
}
```

## Evaluating Agent Runs

When an agent completes a run, TimeSeriesGym creates a run group directory in `runs/` containing:

- Per-competition subdirectories with logs, code, and submissions, the format contract the
  agent's `validate_submission` checked against (`validation_contract.json`), and the run's
  grading (`grading/`)
- `metadata.json` summarizing all runs

### Grading Workflow

The agent's container has the challenge's public data only (at `/home/data`), and the format
check `validate_submission`, which compares a submission with the sample submission and gives no
score. When the agent's container has stopped and its outputs have been extracted,
`run_agent.py` grades the run in a separate container of the grader image: it is the only
container with the challenge's private data (read-only), it has no network, runs as the invoking
user and takes the memory and CPU limits, but not the runtime, GPUs or privileges, of the
container configuration. It runs `timeseriesgym grade` on the line that
`experiments/make_submission.py` writes for the run, and saves the report and its log to
`<run>/grading/`. `metadata.json` records the run's competition report as `grading` (or
`{"error": ...}` if grading failed).

The report of a whole run group comes from the same two steps as before, which also work with
`--skip-grading`:

1. **Create submission file**:
   ```bash
   python experiments/make_submission.py --metadata runs/<run-group>/metadata.json --output runs/<run-group>/submission.jsonl
   ```

2. **Grade submission**:
   ```bash
   timeseriesgym grade --submission runs/<run-group>/submission.jsonl --output-dir runs/<run-group>
   ```

## Agent Details

### Dummy Agent

The dummy agent verifies environment correctness by performing diagnostics:

- Validates runtime permissions
- Checks Python interpreter configuration
- Tests sample submission access
- Checks that its container has no private data
- Confirms proper directory access permissions
- Runs the format check on its submission

### Security Considerations

TimeSeriesGym prioritizes security when running agent code:

- **No private data or graders in the agent's container**: the answers are never mounted in it
  or copied to it, so neither a root process nor Docker-in-Docker in it can read them. They are
  used only by the grading step, after the agent's container has stopped. A privileged container
  can still reach the host's devices, and through them the host's copy of the data: prefer
  Sysbox, which runs Docker-in-Docker without `--privileged`.
- **Sysbox Runtime**: Recommended for AIDE, MLAgentBench, and dummy agents to ensure container isolation
- **OpenHands Execution**: Requires privileged mode due to Docker-in-Docker execution. When running OpenHands, set:
  ```bash
  export I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS=True
  ```

#### Known Limitations

- Sysbox has [known issues](https://github.com/nestybox/sysbox/issues/50) with multiple NVIDIA GPU passthroughs in Docker-in-Docker scenarios
- For GPU-intensive workloads with OpenHands, standard Docker runtime may be required

## Migrating from the Grading Server

Earlier versions mounted the challenge's private data in the agent's container (under `/private`,
protected by a mode-700 directory) and ran a grading server there, at
`http://localhost:5000/validate`. Neither exists any more. To migrate:

- Rebuild `timeseriesgym-env`, build `timeseriesgym-grader`, and then rebuild every agent image
  on the new base. `run_agent.py` refuses an agent image built on an older `timeseriesgym-env`
  (it has no `/opt/timeseriesgym`), and stops before any run if the grader image is missing.
- Agents check their submission with `validate_submission [file or folder]`, or
  `/home/validate_submission.sh` as before. It compares the submission with the sample submission
  instead of running the grader on the answers, prints "Submission is valid. ..." or
  "Submission invalid! <reason>" as plain text (the server returned JSON), and exits with code 1
  for an invalid submission. Replace calls to the HTTP endpoint with the command.
- Each run is graded when its container has stopped. The manual workflow
  (`experiments/make_submission.py`, then `timeseriesgym grade`) is unchanged; pass
  `--skip-grading` to use only that.
- Challenges whose sample submission's rows or id columns do not describe a valid submission
  need `submission_format` in their `config.yaml` (see
  [documentation/adding_new_challenges.md](../documentation/adding_new_challenges.md)).

## Troubleshooting

- If an agent fails to complete, check the logs in `runs/<run-group>/<competition>/logs/`
- For container issues, verify Sysbox installation with `sysbox-runc --version`
- If GPU passthrough fails, try using `"runtime": "docker"` in your container config

## Performance Considerations

- Agent performance varies significantly with hardware resources
- For large-scale evaluations, consider using competition subsets like `lite.txt`
- GPU acceleration is particularly important for LLM-based agents like OpenHands
