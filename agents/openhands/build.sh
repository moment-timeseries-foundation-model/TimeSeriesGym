#!/bin/bash

# Runs as root through sudo (see the Dockerfile) and builds from /home/agent, which nonroot can
# write to. start.sh runs it before the agent starts; later calls do nothing, so files the agent
# changes in /home/agent are never run as root.
if ! mkdir /root/.agent_build_started 2>/dev/null; then
  echo "agent_build.sh has already run in this container; skipping."
  exit 0
fi

cd /home/agent
/opt/conda/bin/conda run -n agent --no-capture-output make build
sudo -u nonroot /opt/conda/bin/conda run -n agent --no-capture-output playwright install
