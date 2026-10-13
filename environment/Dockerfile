# Two images. The agent's base image, timeseriesgym-env, is the default target:
#   docker build --platform=linux/amd64 -t timeseriesgym-env -f environment/Dockerfile .
# The grader grades each run in a separate container after the agent's container has stopped
# (agents/grading.py):
#   docker build --platform=linux/amd64 --target grader -t timeseriesgym-grader \
#     -f environment/Dockerfile .
# Only the grader has TimeSeriesGym and its graders installed, and only the grader's container
# gets the private data. The agent's container gets the public data and a format check.

FROM python:3.11-slim-bookworm AS grader

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES

COPY . /timeseriesgym
RUN pip install -e /timeseriesgym

# run_agent.py runs `timeseriesgym grade` in it as the invoking user, with no network
WORKDIR /tmp
ENTRYPOINT ["timeseriesgym"]

# The agent's base image (the default target)
FROM ubuntu:20.04

# Avoid interactive dialog from apt-get and other packages requiring configuration
ENV DEBIAN_FRONTEND=noninteractive

# install basic packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    unzip \
    zip \
    p7zip-full \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python-is-python3 \
    build-essential \
    openssh-server \
    tmux \
    gettext \
    sudo \
    ffmpeg \
    libsm6 \
    libxext6 \
    && pip install jupyter \
    && rm -rf /var/lib/apt/lists/* # removes cache

RUN pip install virtualenv
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
# NOTE: Try using the appropriate Miniconda version for ARM:
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /opt/conda
RUN rm /tmp/miniconda.sh
RUN /opt/conda/bin/conda init

ARG CONDA_ENV_NAME=agent
ARG PYTHON_VERSION=3.11
ARG REQUIREMENTS=/tmp/requirements.txt

COPY environment/requirements.txt ${REQUIREMENTS}

# create conda environment and optionally install the requirements to it
RUN /opt/conda/bin/conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
ARG INSTALL_HEAVY_DEPENDENCIES=true
ENV INSTALL_HEAVY_DEPENDENCIES=${INSTALL_HEAVY_DEPENDENCIES}

# The rest of your Dockerfile
# NOTE: This does not work yet, skipping
RUN if [ "$INSTALL_HEAVY_DEPENDENCIES" = "true" ]; then \
    /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install -r /tmp/requirements.txt && \
    /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install tensorflow[and-cuda]==2.17 && \
    /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install torch==2.2.0 torchaudio==2.2.0 torchtext==0.17.0 torchvision==0.17.0 && \
    /opt/conda/bin/conda clean -afy ; fi

ENV PATH="/opt/conda/bin:${PATH}"

# Installs from here onward go into the conda base env; previous was installed to /usr/bin/python

# The format check, validate_submission, runs in the conda base env and needs only these. This
# image has no TimeSeriesGym code, graders or private data (see the grader target above).
RUN pip install "numpy>=1.26" "pandas>=2.2" "h5py>=3.10"

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=

# Copy over relevant files. run_agent.py adds the challenge's format, contract.json, to
# /opt/timeseriesgym in each container.
COPY environment/validate_submission.py /opt/timeseriesgym/validate_submission.py
COPY environment/instructions.txt /home/instructions.txt
COPY environment/validate_submission.sh /home/validate_submission.sh
COPY environment/validate_submission.sh /usr/local/bin/validate_submission
COPY environment/entrypoint.sh /entrypoint.sh

# Create nonroot user; make entrypoint and format check executable
RUN useradd -m nonroot \
    && mkdir /home/submission \
    && chmod 755 /opt/timeseriesgym \
    && chmod +x /entrypoint.sh /home/validate_submission.sh /usr/local/bin/validate_submission

WORKDIR /home

# IMPORTANT: This needs to run as root! Downstream Dockerfiles must not change the default USER for when the container starts.
# Entrypoint script is in charge of setting up the user environment (permissions under /home)
ENTRYPOINT ["/entrypoint.sh"]
