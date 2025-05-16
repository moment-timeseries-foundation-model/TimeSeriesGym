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

- Per-competition subdirectories with logs, code, and submissions
- `metadata.json` summarizing all runs

### Grading Workflow

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
- Verifies private data isolation
- Confirms proper directory access permissions

### Security Considerations

TimeSeriesGym prioritizes security when running agent code:

- **Sysbox Runtime**: Recommended for AIDE, MLAgentBench, and dummy agents to ensure container isolation
- **OpenHands Execution**: Requires privileged mode due to Docker-in-Docker execution. When running OpenHands, set:
  ```bash
  export I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS=True
  ```

#### Known Limitations

- Sysbox has [known issues](https://github.com/nestybox/sysbox/issues/50) with multiple NVIDIA GPU passthroughs in Docker-in-Docker scenarios
- For GPU-intensive workloads with OpenHands, standard Docker runtime may be required

## Troubleshooting

- If an agent fails to complete, check the logs in `runs/<run-group>/<competition>/logs/`
- For container issues, verify Sysbox installation with `sysbox-runc --version`
- If GPU passthrough fails, try using `"runtime": "docker"` in your container config

## Performance Considerations

- Agent performance varies significantly with hardware resources
- For large-scale evaluations, consider using competition subsets like `lite.txt`
- GPU acceleration is particularly important for LLM-based agents like OpenHands
