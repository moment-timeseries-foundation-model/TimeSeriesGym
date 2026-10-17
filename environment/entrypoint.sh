#!/bin/bash

# Print commands and their arguments as they are executed
set -x

{
  # log into /home/logs
  LOGS_DIR=/home/logs
  mkdir -p $LOGS_DIR

  # chmod the /home directory such that nonroot users can work on everything within it. We do this at container start
  # time so that anything added later in agent-specific Dockerfiles will also receive the correct permissions.
  # (this command does `chmod a+rw /home` but with the exception of /home/data, which is a read-only volume)
  find /home -path /home/data -prune -o -exec chmod a+rw {} \;
  ls -l /home

  # /home is ready: run_agent.py waits for this file before it starts the agent (agents/run.py)
  touch /run/timeseriesgym-ready

  # Nothing else runs next to the agent: this container has no private data and no grader. The
  # run is graded in a separate container after this one has stopped (agents/grading.py).
  # Stay alive while run_agent.py runs the agent's start.sh.
  sleep infinity
} 2>&1 | tee $LOGS_DIR/entrypoint.log
