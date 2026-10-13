"""Agents' network and API keys: what runs get, and what agent configs may hold.

The model proxy itself is tested in test_model_proxy.py; the network of a real container in
tests/integration/test_agent_network.py.
"""

import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agents import model_proxy as mp
from tests.unit.test_model_proxy import HOST_KEYS, UnixHTTPConnection, _config

REPO = Path(__file__).parents[2]

# ---------------------------------------------------------------- runs and their model proxy


class _FakeExecAPI:
    def __init__(self):
        self.created = []  # the agent's start.sh

    def exec_create(self, container_id, cmd, user):
        self.created.append(cmd)
        return {"Id": "e1"}

    def exec_start(self, exec_id, stream):
        yield b"agent output\n"

    def exec_inspect(self, exec_id):
        return {"ExitCode": 0}


class _FakeContainer:
    """Records its execs; the relay's wait calls `relay_wait` and exits `relay_exit`."""

    def __init__(self, relay_wait=None, relay_exit=0):
        self.id, self.name, self.execs, self.stopped = "c1", "competition-x", [], False
        self.relay_wait, self.relay_exit = relay_wait, relay_exit
        self.client = SimpleNamespace(api=_FakeExecAPI())

    def start(self):
        pass

    def exec_run(self, cmd, **kwargs):
        self.execs.append((cmd, kwargs))
        if tuple(cmd) == mp.ModelProxy.relay_wait_command:
            if self.relay_wait is not None:
                self.relay_wait()
            return (self.relay_exit, b"")
        return (0, b"")  # the wait for the container's entrypoint

    def stop(self):
        self.stopped = True

    def remove(self):
        pass


def _agent(model_proxy=None, env_vars=None):
    return SimpleNamespace(
        kwargs_type=None,
        kwargs={},
        env_vars=env_vars or {},
        privileged=False,
        model_proxy=model_proxy,
    )


def _run(tmp_path, monkeypatch, agent, container=None, created=None, **options):
    agents_run = importlib.import_module("agents.run")
    container = container or _FakeContainer()
    created = [] if created is None else created

    def create(**kw):
        created.append(kw)
        return container

    monkeypatch.setattr(agents_run, "create_competition_container", create)
    monkeypatch.setattr(agents_run, "save_output", lambda c, run_dir, cfg: None)
    competition = SimpleNamespace(id="x", public_dir=tmp_path, private_dir=tmp_path)
    exit_code = agents_run.run_in_container(
        client=None,
        competition=competition,
        agent=agent,
        image="img",
        container_config={},
        retain_container=False,
        run_dir=tmp_path,
        logger=logging.getLogger("test"),
        **options,
    )
    assert exit_code == 0
    return created[0], container


def _commands(container) -> list[tuple]:
    return [tuple(cmd) if isinstance(cmd, list) else (cmd,) for cmd, _ in container.execs]


def _proxy_mount(created: dict) -> tuple[str, dict]:
    volumes = created["volumes_config"].items()
    return next((k, v) for k, v in volumes if v["bind"] == mp.CONTAINER_DIR)


def test_runs_reach_their_models_through_their_proxy(tmp_path, monkeypatch):
    agent = _agent(model_proxy=_config(monkeypatch), env_vars={"X": "1"})
    created, statuses = [], []

    def health():  # while the run is on, the mounted socket answers
        conn = UnixHTTPConnection(Path(_proxy_mount(created[0])[0]) / mp.SOCKET_NAME)
        conn.request("GET", "/health")
        statuses.append(conn.getresponse().status)
        conn.close()

    created, container = _run(
        tmp_path, monkeypatch, agent, container=_FakeContainer(health), created=created
    )
    assert statuses == [200]
    env = created["env_vars"]
    assert env["X"] == "1" and env["OPENAI_API_KEY"] == mp.DUMMY_KEY
    assert env["OPENAI_BASE_URL"].startswith(f"http://127.0.0.1:{mp.RELAY_PORT}/t/")
    assert HOST_KEYS["openai"] not in json.dumps(env)
    sock_dir, mount = _proxy_mount(created)
    assert mount["mode"] == "ro"
    # after the entrypoint's wait: the relay as `nobody`, then the wait for it
    commands, options = _commands(container), [kw for _, kw in container.execs]
    start = commands.index(mp.ModelProxy.relay_command)
    assert start == 1 and options[start] == {"user": "nobody", "detach": True}
    assert commands[start + 1] == mp.ModelProxy.relay_wait_command
    assert not Path(sock_dir).exists()  # the proxy ended with the run
    assert (tmp_path / mp.LOG_NAME).exists()


def test_a_run_whose_relay_does_not_answer_fails_and_cleans_up(tmp_path, monkeypatch):
    agent = _agent(model_proxy=_config(monkeypatch))
    container, created = _FakeContainer(relay_exit=1), []
    with pytest.raises(RuntimeError, match="relay"):
        _run(tmp_path, monkeypatch, agent, container=container, created=created)
    assert container.stopped and not Path(_proxy_mount(created[0])[0]).exists()
    assert container.client.api.created == []  # the agent never started


def test_a_run_whose_container_cannot_be_created_closes_its_proxy(tmp_path, monkeypatch):
    agents_run = importlib.import_module("agents.run")
    made = []
    real = mp.ModelProxy.for_agent

    def for_agent(agent, run_dir):
        made.append(real(agent, run_dir))
        return made[-1]

    def create(**kw):
        raise RuntimeError("docker said no")

    monkeypatch.setattr(agents_run.ModelProxy, "for_agent", for_agent)
    monkeypatch.setattr(agents_run, "create_competition_container", create)
    competition = SimpleNamespace(id="x", public_dir=tmp_path, private_dir=tmp_path)
    with pytest.raises(RuntimeError, match="docker said no"):
        agents_run.run_in_container(
            client=None,
            competition=competition,
            agent=_agent(model_proxy=_config(monkeypatch)),
            image="img",
            container_config={},
            retain_container=False,
            run_dir=tmp_path,
            logger=logging.getLogger("test"),
        )
    assert not made[0].dir.exists()


def test_the_agents_own_variables_cannot_redirect_its_model_calls(tmp_path, monkeypatch):
    agent = _agent(_config(monkeypatch), env_vars={"OPENAI_BASE_URL": "https://api.openai.com"})
    created, _ = _run(tmp_path, monkeypatch, agent)
    assert created["env_vars"]["OPENAI_BASE_URL"].startswith("http://127.0.0.1:")


def test_runs_without_a_proxy_config_get_none(tmp_path, monkeypatch):
    created, container = _run(tmp_path, monkeypatch, _agent())
    assert not any("OPENAI" in k for k in created["env_vars"])
    assert mp.ModelProxy.relay_command not in _commands(container)
    assert not (tmp_path / mp.LOG_NAME).exists()


# ---------------------------------------------------------------- agent configs


def _registry_with(tmp_path, monkeypatch, entry):
    from agents.registry import Registry

    agent_dir = tmp_path / "agents" / "x"
    agent_dir.mkdir(parents=True)
    (agent_dir / "start.sh").write_text("")
    (agent_dir / "Dockerfile").write_text("")
    (agent_dir / "config.yaml").write_text(
        yaml.safe_dump({"x": {"start": "x/start.sh", "dockerfile": "x/Dockerfile", **entry}})
    )
    monkeypatch.setattr(Registry, "get_agents_dir", lambda self: tmp_path / "agents")
    return Registry()


def test_an_agents_proxy_config_is_parsed_and_never_printed(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", HOST_KEYS["openai"])
    proxy = {"providers": {"openai": {"key": "${{ secrets.TEST_OPENAI_KEY }}", "models": ["m"]}}}
    agent = _registry_with(tmp_path, monkeypatch, {"model_proxy": proxy}).get_agent("x")
    assert agent.model_proxy["keys"] == {"openai": HOST_KEYS["openai"]}
    assert HOST_KEYS["openai"] not in repr(agent)
    assert all(HOST_KEYS["openai"] not in str(v) for v in agent.env_vars.values())
