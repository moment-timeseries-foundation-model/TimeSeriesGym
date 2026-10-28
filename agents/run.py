import json
import logging
import string
import time
from pathlib import Path

import docker
from docker.models.containers import Container
from dotenv import dotenv_values

from agents.model_proxy import ModelProxy
from agents.registry import Agent
from agents.validation import install_contract, submission_contract
from environment.utils import (
    create_competition_container,
    extract_from_container,
    extract_from_container_sysbox,
    put_file,
)
from timeseriesgym.registry import Competition
from timeseriesgym.utils import purple

CONSTANTS = dotenv_values(Path(__file__).parent.resolve() / ".shared_env")
DATA_DIR = "/home/data"  # the competition's public data, in the agent's container
READY_FILE = "/run/timeseriesgym-ready"  # environment/entrypoint.sh: /home is ready for the agent
INSTRUCTIONS = Path(__file__).parents[1] / "environment" / "instructions.txt"
INSTRUCTIONS_PATH = "/home/instructions.txt"  # in the agent's container
NO_NETWORK = (
    "- This container has no internet access: nothing can be downloaded or installed. Use the "
    "packages that are installed and the data in `/home/data/`"
)


def instructions(network: str) -> str:
    """The agents' general instructions (environment/instructions.txt), with the line on the
    network of their container: "none", or the Docker network run_agent.py was given."""
    line = NO_NETWORK
    if network != "none":
        line = f"- This container is on the Docker network `{network}`: it may reach the internet"
    return string.Template(INSTRUCTIONS.read_text()).substitute(network_access=line)


def check_privileges(agent: Agent, container_config: dict, allow_without_sysbox: bool) -> None:
    """A privileged agent container can reach the host's devices, and through them the host's
    copy of the private data, unless it runs under Sysbox (`"runtime": "sysbox-runc"` in the
    container configuration), which keeps it in a user namespace. Raises ValueError for a
    privileged agent without Sysbox, unless `allow_without_sysbox`."""
    privileged = agent.privileged or bool(container_config.get("privileged"))
    if privileged and container_config.get("runtime") != "sysbox-runc" and not allow_without_sysbox:
        raise ValueError(
            "The agent runs in a privileged container, and the container configuration does not "
            'use Sysbox ("runtime": "sysbox-runc"): agent code could reach the host and the '
            "private data there. Use a configuration with Sysbox, or pass "
            "--allow-privileged-without-sysbox to accept this (see agents/README.md)."
        )


def save_output(container: Container, save_dir: Path, container_config: dict) -> Path:
    """
    Extracts the submission, logs, and code directories from the container

    and saves them to the specified directory.

    Args:
        container: The Docker container.
        save_dir: The directory where the output file will be saved.
        container_config: The container configuration.
    Returns:
        Path to the output directory.
    """
    if "runtime" in container_config and container_config["runtime"] == "sysbox-runc":
        extraction_fn = extract_from_container_sysbox
    else:
        extraction_fn = extract_from_container

    for dir_type in ["SUBMISSION_DIR", "LOGS_DIR", "CODE_DIR"]:
        container_dir = CONSTANTS[dir_type]
        extraction_fn(container, container_dir, save_dir)

    return save_dir


def execute_agent(container: Container, agent: Agent, logger: logging.Logger) -> int:
    """
    Initiates the agent via its start script inside the container and returns its exit code.
    """
    cmd = ["bash", f"{CONSTANTS['AGENT_DIR']}/start.sh"]

    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]

    if agent.kwargs_type == "omegaconf":
        for key, value in agent.kwargs.items():
            if value is None:
                cmd.append(f"{key}=null")
            else:
                cmd.append(f"{key}={value}")

    logger.info("Running agent...")
    # what container.exec_run(cmd, stream=True) does, except that it returns no exit code
    api = container.client.api
    exec_id = api.exec_create(container.id, cmd, user="nonroot")["Id"]

    for chunk in api.exec_start(exec_id, stream=True):
        logger.info(f"[Container] {chunk.decode('utf-8').strip()}")

    return api.exec_inspect(exec_id)["ExitCode"]


def clean_up(container: Container, logger: logging.Logger, retain: bool = False) -> bool:
    """
    Stops and removes the container.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Cleaning up container: {container.name}")
    try:
        container.stop()
        if not retain:
            container.remove()
        logger.info(f"Container {container.name} stopped and removed.")
        return True
    except Exception as e:
        logger.error(
            f"Error cleaning up: {e}. You may wish to manually check the status of the "
            f"{container.name} container."
        )
        return False


def run_in_container(
    client: docker.DockerClient,
    competition: Competition,
    agent: Agent,
    image: str,
    container_config: dict,
    retain_container: bool,
    run_dir: Path,
    logger: logging.Logger,
    network: str = "none",
) -> int:
    """
    Runs the agent in a container with the competition's public data (no private data, no
    grader: the run is graded after this container has stopped, see agents/grading.py) and
    the contract of its format check, `validate_submission`. The outputs are saved to
    `run_dir` whatever the agent's exit code.

    The agents' instructions are written into the container with a line on its network.

    The container is on the Docker network `network`; the default, "none", gives it no network.
    An agent with a `model_proxy` config calls its models through a model proxy for this run
    (agents/model_proxy.py), which keeps the API keys on the host.

    Args:
        client: Docker client.
        competition: The competition to run.
        agent: The agent to run.
        image: The Docker image to use. Assumes the image is built.
        container_config: Configuration for the Docker container.
        retain_container: Whether to retain the container after the run instead of removing it.
        run_dir: Path to the directory where all assets associated with the run are stored.
        logger: Logger for the run.
        network: Docker network mode of the container.

    Returns:
        The exit code of the agent's start script.
    """
    volumes_config = {
        competition.public_dir.resolve().as_posix(): {
            "bind": DATA_DIR,
            "mode": "ro",
        },
    }
    contract = submission_contract(
        competition,
        data_dir=DATA_DIR,
        submission_dir=CONSTANTS["SUBMISSION_DIR"],
        code_dir=CONSTANTS["CODE_DIR"],
    )
    (run_dir / "validation_contract.json").write_text(json.dumps(contract, indent=2))

    proxy = ModelProxy.for_agent(agent, run_dir)  # None for an agent without a model_proxy
    try:
        container = create_competition_container(
            client=client,
            competition=competition,
            container_config=container_config,
            volumes_config={**volumes_config, **(proxy.volumes if proxy else {})},
            env_vars={
                "COMPETITION_ID": competition.id,
                **agent.env_vars,
                **(proxy.container_env() if proxy else {}),  # after the agent's own: these win
            },
            container_image=image,
            privileged=agent.privileged,
            network_mode=network,
        )
    except BaseException:
        if proxy is not None:
            proxy.close()
        raise

    logger.info(purple(f"Run started: {run_dir}"))
    try:
        time_start = time.monotonic()
        install_contract(container, contract)
        put_file(container, INSTRUCTIONS_PATH, instructions(network).encode())
        container.start()
        exit_code, _ = container.exec_run(
            ["timeout", "60", "sh", "-c", f"while [ ! -e {READY_FILE} ]; do sleep 1; done"]
        )
        if exit_code != 0:
            raise RuntimeError(
                "The container's entrypoint did not finish within 60 seconds (it makes /home "
                "writable for the agent). This is likely due to an error in `entrypoint.sh`; "
                "check the logs."
            )
        if proxy is not None:
            proxy.start_relay(container)  # and waits until the proxy answers through it
        exit_code = execute_agent(container, agent, logger)
        # also for failed runs: their logs, and the submission an agent that timed out had saved
        save_output(container, run_dir, container_config)
        time_end = time.monotonic()
        if exit_code != 0:
            logger.warning(f"The agent's start script exited with code {exit_code}.")
        logger.info(f"Run completed in {time_end - time_start:.2f} seconds.")
        return exit_code
    finally:
        clean_up(container, logger, retain_container)
        if proxy is not None:
            proxy.close()
            logger.info(proxy.summary())
