#!/bin/bash

# Print commands and their arguments as they are executed
set -x

# run the root entrypoint in the background
/entrypoint.sh &

mkdir -p $LOGS_DIR
mkdir -p $AGENT_DIR
{
  # Check if Docker installed, and if so start the Docker daemon in the background.
  if [ -x "$(command -v docker)" ]; then

    # if CUDA is available, let Docker use it (the image has the nvidia container toolkit)
    if [ -x "$(command -v nvidia-smi)" ]; then
      sudo nvidia-ctk runtime configure --runtime=docker
    fi

    # (re)start the Docker daemon
    if sudo pgrep dockerd > /dev/null; then
      sudo pkill dockerd
    fi
    sudo dockerd > $LOGS_DIR/docker.log 2>&1 &
    sleep 5
  else
    echo "Docker not installed. Skipping Docker startup."
  fi

} 2>&1 | tee $LOGS_DIR/agent_entrypoint.log

# signal that the entrypoint has finished
touch $AGENT_DIR/entrypoint_done

# wait for root entrypoint (a server), need this otherwise the container exits
wait
