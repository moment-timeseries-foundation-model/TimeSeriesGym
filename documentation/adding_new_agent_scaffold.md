# Adding a New Agent Scaffold to TimeSeriesGym

This guide explains how to integrate a new agent scaffold into TimeSeriesGym, allowing you to evaluate additional AI agent architectures on time series machine learning engineering tasks.

## Overview

TimeSeriesGym is designed to support multiple agent architectures, each with different approaches to solving machine learning engineering challenges. Adding a new agent scaffold involves creating the necessary Docker infrastructure, implementing the agent interface, and ensuring proper integration with the evaluation framework.

## Step 1: Create the Agent Directory Structure

Start by creating a dedicated directory for your agent:

```bash
mkdir -p agents/<agent-id>/
```

Where `<agent-id>` is a unique identifier for your agent (e.g., `my-agent`).

## Step 2: Create a Dockerfile

Create a `Dockerfile` in your agent directory that builds upon the base TimeSeriesGym environment. Here's an example based on the AIDE agent implementation:

```dockerfile
# agents/<agent-id>/Dockerfile
FROM timeseriesgym-env

# Set environment variables for directory structure (passed at build time)
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

# Create directories
RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

# Define conda environment name (if using conda)
ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# Copy just the requirements file for better Docker layer caching
COPY requirements.txt ${AGENT_DIR}/requirements.txt

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in conda environment
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# Copy agent code into the container
COPY . ${AGENT_DIR}

# Set the working directory
WORKDIR ${AGENT_DIR}

# Set entrypoint to start script
ENTRYPOINT ["./start.sh"]
```

## Step 3: Create a Start Script

Create a `start.sh` script that handles environment setup and executes your agent:

```bash
#!/bin/bash
set -x  # Print commands as they are executed

cd ${AGENT_DIR}

# Activate conda environment if using conda
eval "$(conda shell.bash hook)"
conda activate agent

# Detect available hardware
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE

# Test GPU availability for common frameworks
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# Format time limit for readable output
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# Prepare agent instructions by combining general and competition-specific instructions
cp /home/instructions.txt ${AGENT_DIR}/full_instructions.txt
# Make paths relative for the agent
sed -i 's|/home/||g' ${AGENT_DIR}/full_instructions.txt

# Add agent-specific instructions
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt

# Add competition instructions
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt
cat /home/data/description.md >> ${AGENT_DIR}/full_instructions.txt

# Set up symbolic links to the expected output directories
mkdir -p ${AGENT_DIR}/workspaces/exp
mkdir -p ${AGENT_DIR}/logs
ln -s ${LOGS_DIR} ${AGENT_DIR}/logs/exp
ln -s ${CODE_DIR} ${AGENT_DIR}/workspaces/exp/best_solution
ln -s ${SUBMISSION_DIR} ${AGENT_DIR}/workspaces/exp/best_submission

# Run the agent with timeout
timeout $TIME_LIMIT_SECS python main.py \
  --data_dir="/home/data/" \
  --desc_file="${AGENT_DIR}/full_instructions.txt" \
  --exp_name="exp" \
  $@  # Forward any additional arguments

# Check if timeout occurred
if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi
```

Make the script executable:

```bash
chmod +x agents/<agent-id>/start.sh
```

## Step 4: Create a Configuration File

Create a `config.yaml` file to specify agent parameters and resource requirements:

```yaml
# agents/<agent-id>/config.yaml
vars:
  step_count: &step_count 50
  time_limit: &time_limit 14400  # 4 hours in seconds

defaults: &defaults
  start: <agent-id>/start.sh
  dockerfile: <agent-id>/Dockerfile
  kwargs_type: omegaconf
  env_vars: &env_vars
    TIME_LIMIT_SECS: *time_limit
    STEP_LIMIT: *step_count

kwargs_common: &kwargs_common
  agent.search.max_debug_depth: 20
  agent.search.debug_prob: 1
  agent.time_limit: *time_limit
  exec.timeout: 32400  # 9 hours limit per step
  copy_data: False  # use symbolic links
  agent.cost_limit: null
  agent.remind_resource_limit: True

<agent-id>:
  <<: *defaults
  kwargs:
    <<: *kwargs_common
    # Agent-specific configuration
    agent.model: gpt-4
    agent.steps: *step_count
  env_vars:
    <<: *env_vars
    # Optional: API keys or other environment variables
    API_KEY: ${{ secrets.API_KEY }}
```

## Step 5: Create Additional Agent Files

### requirements.txt

Create a `requirements.txt` file with the dependencies your agent needs:

```
# agents/<agent-id>/requirements.txt
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.9.0
tensorflow>=2.6.0
transformers>=4.12.0
# Add any other dependencies
```

### additional_notes.txt

Create an `additional_notes.txt` file with agent-specific instructions:

```
# agents/<agent-id>/additional_notes.txt
AGENT INSTRUCTIONS
------

You are running on ${HARDWARE}.
You have ${TIME_LIMIT} to complete this task.

Your task is to solve the machine learning competition described below. Follow these steps:

1. Analyze the provided data files
2. Develop a solution approach
3. Implement your solution
4. Create a submission file according to the competition requirements

Your submission will be saved to the 'best_submission' directory.
Your code will be saved to the 'best_solution' directory.

Good luck!
```

## Step 6: Update Agent Registry

Add your agent to the registry by updating the appropriate files in TimeSeriesGym:

1. Update `run_agent.py` to include your agent ID
2. Add your agent's container configuration settings

## Step 7: Building and Testing Your Agent

Build your agent's Docker image:

```bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

docker build --platform=linux/amd64 -t <agent-id> agents/<agent-id>/ \
--build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
--build-arg LOGS_DIR=$LOGS_DIR \
--build-arg CODE_DIR=$CODE_DIR \
--build-arg AGENT_DIR=$AGENT_DIR
```

Test your agent on a development competition:

```bash
python run_agent.py --agent-id <agent-id> --competition-set experiments/splits/dev.txt
```

## Example: AIDE Agent Integration

The AIDE agent provides a good reference implementation:

### Directory Structure

```
agents/aide/
├── Dockerfile             # Docker configuration
├── start.sh               # Startup script
├── config.yaml            # Agent configuration
├── requirements.txt       # Python dependencies
├── additional_notes.txt   # Agent-specific instructions
└── main.py                # Core agent implementation
```

### Execution Flow

1. TimeSeriesGym builds the agent's Docker image
2. Competition data is mounted to `/home/data/`
3. `start.sh` is executed, which:
   - Sets up the environment
   - Prepares instructions
   - Creates symbolic links for output directories
   - Launches the agent with appropriate parameters
4. The agent processes the competition and generates a submission
5. Output files are collected from the linked directories

## Advanced Integration Considerations

### Resource Management

Configure resource limits in the container configuration:

```yaml
# Container configuration
container_config = {
    "memory": "16g",
    "cpus": "8",
    "runtime": "sysbox-runc",  # Use sysbox for security
    "gpus": "0,1",  # If GPU support is needed
}
```

### Security Considerations

For agents requiring privileged access:

```python
# In run_agent.py
if agent_id == "<agent-id>" and os.environ.get("I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS") != "True":
    raise ValueError("This agent requires privileged mode. Set I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS=True")
```

## Conclusion

By following this guide, you can integrate a new agent scaffold into TimeSeriesGym. The key components—Dockerfile, start script, configuration file, and core implementation—provide the framework needed for your agent to work within the TimeSeriesGym evaluation environment.

Remember to test your agent thoroughly on various competition types to ensure it can handle different time series machine learning engineering challenges.
