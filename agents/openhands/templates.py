from string import Template

config_template = Template(
    """
[core]

# Base path for the workspace
workspace_base="${workspace_base}"

# Debugging enabled
debug="${debug}"

# Path to store trajectories, can be a folder or a file
# If it's a folder, the session id will be used as the file name
save_trajectory_path="./trajectories"

# Maximum number of iterations
max_iterations=${max_steps}

# Path to mount the workspace in the sandbox
workspace_mount_path_in_sandbox="${workspace_mount_path_in_sandbox}"

# Path to mount the workspace
workspace_mount_path="${workspace_base}"

# Name of the default agent
default_agent="${agent}"

# Whether to enable the default LLM summarizing condenser when no condenser is specified in config
# When true, a LLMSummarizingCondenserConfig will be used as the default condenser
# When false, a NoOpCondenserConfig (no summarization) will be used
enable_default_condenser="${enable_default_condenser}"

[llm]
# API key to use
api_key="${api_key}"

# Cost per input token
input_cost_per_token="${input_cost_per_token}"

# Cost per output token
output_cost_per_token="${output_cost_per_token}"

# Model to use.
model="${model}"

# Using the prompt caching feature if provided by the LLM and supported
caching_prompt="${caching_prompt}"

[agent]

# Whether the browsing tool is enabled
enable_browsing="${enable_browsing}"

[sandbox]

# Sandbox timeout in seconds
timeout=600

# Use host network
use_host_network=false

[sandbox.docker_runtime_kwargs]
shm_size="${shm_size}"
"""
)

# TODO(mononito): Change this to account for TimeSeriesGym
# ruff: noqa: E501
additional_notes_template = Template(
    """
ADDITIONAL NOTES
------
- **Compute**: You have access to a ${type_of_processor} with the appropriate drivers installed.
- **Total Runtime**: You have a maximum of ${max_time_in_hours} hours to attempt this task. Bear in mind that program runtime also counts towards this limit. TRY TO USE AS MUCH OF THE AVAILABLE TIME AS POSSIBLE to iteratively develop your solution.
- **Total Steps**: You have a maximum of ${max_steps} steps to submit your solution.
- **Deliverables**: Carefully read the instructions in `/home/data/description.md`. Save **ALL** the code that you generate to `/home/code`. This could be `.py` files as well as `.ipynb` notebooks.
- You only have read and write permissions to the files that you create. You therefore cannot use the helper script `/home/validate_submission.sh`. Instead, use the HTTP endpoint (e.g. via `curl -X POST -F "file=@/path/to/your/file" http://localhost:5000/validate`) to validate your submission.
- You must write all of the code necessary to produce the outputs described in the instructions in `${workspace}`. You must not write to any other location.
- You must use all the time available to you, unless you've achieved the maximum possible score for the competition. Do not stop until you've reached the maximum possible score.
"""
)
