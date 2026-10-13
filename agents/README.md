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

Everything an agent needs at run time must be installed in its image (packages, model weights,
tokenizer files): agents have no network when they run (see below).

## Network and Model Access

Agent containers have no network (Docker's `none` network): agent code can neither download the
public data the challenges are built from (Kaggle, UCR, PhysioNet, Hugging Face, this repository),
nor send anything out. They hold no API keys either. Model calls go through a **model proxy**
(`agents/model_proxy.py`) that `run_agent.py` starts on the host for each run:

- The agent's config names its providers, the API key of each, read from the host's environment,
  and the models the agent may call:

  ```yaml
  model_proxy:
    providers:
      openai:
        key: ${{ secrets.OPENAI_API_KEY }}  # never enters the container
        models: [gpt-4.1-2025-04-14]       # the "model" values the agent may send
    max_requests: 1000                     # optional, per run (default 2000)
    max_seconds: 16200                     # optional, per run (default 24 hours)
  ```

  `${{ secrets.* }}` values anywhere else (`env_vars`, `kwargs`) are refused: agent code can read
  its container's environment and command lines.
- The proxy listens on a unix socket in a new directory, mounted read-only into the container at
  `/run/tsgym-model-proxy`. After the container starts, `run.py` starts a relay in it (as
  `nobody`, so that the agent cannot stop it), which listens on `127.0.0.1:18080` and passes each
  connection to the socket.
- The container gets base URLs that hold a random token of the run, and dummy keys: for each of
  its providers the variables below, `<PROVIDER>_API_KEY=tsgym-model-proxy-dummy-key`, and
  `MODEL_PROXY_URL` (`http://127.0.0.1:18080/t/<token>`, the base of all of them). The SDKs read
  them; values that the agent's own `env_vars` set for these variables are overridden.
- The proxy forwards only `POST` requests to the model endpoints below whose JSON `model` is in the
  agent's list for that provider, adds the host's key, and refuses everything else: other paths,
  methods, providers and models, a wrong token, bodies over 32 MiB. The agent's own auth headers
  are dropped, and a provider's reply to a rejected key is replaced, as it can quote part of the
  key. After `max_requests` forwarded requests or `max_seconds` it answers 429. Connections are
  bounded (timeouts, a request deadline, at most 64 at once, 256 MiB of request bodies in memory).
- `<run>/model_proxy.jsonl` logs counts, status codes, sizes, durations and the allowed model of
  each request; never request or response content, the token or the keys. `run.log` ends with the
  run's totals.

| Provider | Base URL variables | Upstream | Forwarded paths |
|----------|--------------------|----------|-----------------|
| `openai` | `OPENAI_BASE_URL`, `OPENAI_API_BASE` | `https://api.openai.com` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses` |
| `anthropic` | `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | `/v1/messages`, `/v1/messages/count_tokens`, `/v1/complete` |
| `openrouter` | `OPENROUTER_BASE_URL`, `OPENROUTER_API_BASE` | `https://openrouter.ai` | `/api/v1/chat/completions`, `/api/v1/completions` |
| `gemini` | `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | `/v1beta/openai/chat/completions` (Gemini's OpenAI-compatible API) |

A provider is one entry of `PROVIDERS` in `agents/model_proxy.py`.

### Challenges That Need the Network

Some descriptions ask for an install or a download at run time: `moment-anomaly-detection-score`
(`pip install momentfm`, and the MOMENT weights), `context-is-key-moirai` and
`context-is-key-nn5-chronos` (`pip install -r requirements.txt`, and model weights), the two
`time-series-library-*` challenges (`pip install -r requirements.txt`), and the CSDI challenges
(a dataset download; they are not in the full split). Install what they need in the agent's image,
or choose a network for the run group:

```bash
python run_agent.py --agent-id aide --competition-set <split> --agent-network bridge
```

`--agent-network` takes `bridge`, Docker's default network, or the name of a network you created,
for example one that only reaches a package mirror; `host` and `container:<id>` are refused, and
container configs cannot set the network. Agents then reach whatever that network reaches,
including the public answers on `bridge`, so these results are not comparable with isolated runs:
`metadata.json` records the run group's `agent_network`. The model proxy keeps the keys on the host
on every network.

### Limitations

- The proxy's directory is bind-mounted from the host: this needs a Linux Docker host (Docker
  Desktop on macOS does not pass host unix sockets into containers). If `run_agent.py` itself runs
  in a container, `$TMPDIR` must be a host directory mounted there at the same path.
- The relay runs with the image's `python3` (every image built on `timeseriesgym-env` has it).
- The socket is world-writable, and its directory traversable, so that the relay also reaches it
  under Sysbox's user-id mapping; the token guards it. Sysbox was not available to test this.
- A privileged container (OpenHands) can reach the host's devices and leave its network
  namespace: the network isolation holds only for unprivileged containers.
- `tests/integration/test_agent_network.py` checks a real container: no route but loopback, no
  reachable host address, and the proxy reachable only through the relay.

### Migrating Agent Scaffolds

- **Agent configs**: move each `API_KEY: ${{ secrets.X }}` from `env_vars` to
  `model_proxy.providers.<provider>.key`, and list the models the agent's kwargs name.
- **AIDE**: AIDE creates its OpenAI and Anthropic clients with the SDKs' defaults, which read
  `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL`, so the OpenAI variants and
  `aide/claude-3-5-sonnet` need no change to the scaffold. The Claude, Llama and Gemini variants
  now also get the `openai` provider: their feedback model is `gpt-4o-2024-08-06` (the old configs
  passed only the other provider's key). Two variants need a change in the AIDE fork before they
  can run: `aide/llama-3.1-405b-instruct`, as AIDE's OpenRouter client has
  `https://openrouter.ai/api/v1` built in (it must read `OPENROUTER_BASE_URL`), and
  `aide/gemini-1.5-pro`, as AIDE calls Gemini through Google's SDK (it must use Gemini's
  OpenAI-compatible API at `GEMINI_BASE_URL`). Until then their model calls cannot leave the
  container. (This describes AIDE upstream; the fork's source could not be checked here.)
- **MLAgentBench**: `openai==0.28` reads `OPENAI_API_BASE`. The image now downloads tiktoken's
  encodings when it is built (`TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache`). Not verified: that the
  MLAgentBench fork does not set its API base itself.
- **OpenHands**: the image now runs `make build`, `playwright install` (browsers in
  `/opt/ms-playwright`) and installs the NVIDIA container toolkit when it is built; these ran at
  agent start before, through a sudo rule for a build script, which is gone. Its LLM gets the
  proxy as `base_url`, and litellm uses the model price list it ships with. OpenHands' Docker
  runtime still builds its sandbox image when the agent starts (it pulls a base image and installs
  packages), so OpenHands needs `--agent-network bridge` until that image is provided offline, or
  until it uses OpenHands' local runtime.
- **Dummy**: no model calls; nothing changes.
- **New scaffolds**: read the base URLs from the environment (or build them from
  `MODEL_PROXY_URL`), and install everything in the image.

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
