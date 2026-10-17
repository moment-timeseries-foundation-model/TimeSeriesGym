FROM --platform=linux/amd64 timeseriesgym-env

# Do not modify the following block of arg-env pairs. The evaluation infrastructure will
# periodically download the contents of these directories for logging purposes.

# Where the agent should write its `submission.csv`
ARG SUBMISSION_DIR=/home/submission
ENV SUBMISSION_DIR=${SUBMISSION_DIR}

# Where the agent's logs should be written
ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}

# Where the agent should write its code
ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}

# Where the agent code should exist
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG PYTHON_VERSION=3.12
ARG OD_VERSION=0.34.0
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p ${LOGS_DIR} ${AGENT_DIR} ${CODE_DIR} ${SUBMISSION_DIR} && \
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && \
    chmod 700 /tmp/get-docker.sh && \
    /tmp/get-docker.sh && \
    sudo usermod -aG docker nonroot && \
    git clone --branch ${OD_VERSION} --single-branch https://github.com/All-Hands-AI/OpenHands.git ${AGENT_DIR}

WORKDIR ${AGENT_DIR}

# Assumes that the `agent` conda env already exists, which is created in the `timeseriesgym-env` base image.
RUN conda install -y -n ${CONDA_ENV_NAME} -c conda-forge python=3.12 nodejs=22.13.0 && \
    conda run -n ${CONDA_ENV_NAME} pip install poetry>=2.1.2 && \
    conda run -n ${CONDA_ENV_NAME} poetry install && \
    conda run -n ${CONDA_ENV_NAME} poetry add build && \
    conda clean -afy

# Agents have no network at run time (agents/README.md), so everything is installed now:
# OpenHands' build (its Python and frontend dependencies), Playwright's browsers, readable by
# nonroot, and the NVIDIA container toolkit, which entrypoint.sh configures if there is a GPU.
# (These ran at agent start before, through a sudo rule for a build script.)
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
RUN conda run -n ${CONDA_ENV_NAME} --no-capture-output make build && \
    conda run -n ${CONDA_ENV_NAME} --no-capture-output playwright install && \
    chmod -R a+rX ${PLAYWRIGHT_BROWSERS_PATH}
RUN keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    repo=https://nvidia.github.io/libnvidia-container && \
    apt-get update && apt-get install -y gnupg && \
    curl -fsSL $repo/gpgkey | gpg --dearmor -o $keyring && \
    curl -fsSL $repo/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && apt-get install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*

COPY setup.py start.py templates.py utils.py start.sh ${AGENT_DIR}/
COPY entrypoint.sh /agent_entrypoint.sh

RUN chmod +x /agent_entrypoint.sh

# OpenHands logging controls
ENV LOG_ALL_EVENTS=1
# litellm downloads its model price list at import unless it uses the copy it ships with
ENV LITELLM_LOCAL_MODEL_COST_MAP=True

# run the agent's entrypoint script instead (which also runs root entrypoint)
ENTRYPOINT ["/agent_entrypoint.sh"]
