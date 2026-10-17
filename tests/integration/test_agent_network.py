"""Network isolation of agent containers, checked with a throwaway container (needs Docker).

    TSG_NETWORK_TEST_IMAGE=timeseriesgym-env \
      python -m pytest tests/integration/test_agent_network.py

The image must exist (it is never pulled) and have `python3`; any agent image does. Nothing
outside the host is contacted: the proxy's provider is a local stand-in, and the only other
addresses tried are the documentation ranges 192.0.2.0/24 and 2001:db8::/32, which are never
routed, and a listener of the test's own on the gateway of Docker's default bridge.

Run pytest on the Docker host. The proxy's directory (in $TMPDIR) is bind-mounted by the Docker
daemon, so from inside a container with the Docker socket, $TMPDIR must be a host directory
mounted at the same path; and the bridge check needs the host's network namespace.
"""

import dataclasses
import inspect
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import docker
import pytest

from agents import model_proxy as mp
from agents.run import run_in_container

IMAGE = os.environ.get("TSG_NETWORK_TEST_IMAGE", "timeseriesgym-env")
LABEL = {"timeseriesgym-test": "agent-network"}
HOST_KEY = "sk-host-key-for-the-network-test"

PROBE = r"""
import errno, http.client, json, os, socket, sys, urllib.parse

def connect(family, address):
    s = socket.socket(family, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect(address)
        return "connected"
    except OSError as e:
        return errno.errorcode.get(e.errno, type(e).__name__)
    finally:
        s.close()

def post(path, model):
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    conn.request("POST", path, body=json.dumps({"model": model}), headers=headers)
    status = conn.getresponse().status
    conn.close()
    return status

url = urllib.parse.urlsplit(os.environ["OPENAI_BASE_URL"])
targets, isolated = json.loads(sys.argv[1])
with open("/proc/net/route") as f:
    ipv4_routes = f.read().splitlines()[1:]
with open("/proc/net/ipv6_route") as f:
    ipv6_devices = sorted({line.split()[-1] for line in f.read().splitlines()})
print(json.dumps({
    "ipv4_routes": ipv4_routes,
    "ipv6_devices": ipv6_devices,
    # only without a route: elsewhere these would leave the host (unanswered)
    "test_net": isolated and connect(socket.AF_INET, ("192.0.2.1", 80)),
    "test_net6": isolated and connect(socket.AF_INET6, ("2001:db8::1", 80)),
    "targets": [connect(socket.AF_INET, (h, p)) for h, p in targets],
    "allowed": post(url.path + "/chat/completions", "gpt-test"),
    "other_model": post(url.path + "/chat/completions", "gpt-other"),
    "no_token": post("/openai/v1/chat/completions", "gpt-test"),
}))
"""


@pytest.fixture(scope="module")
def client():
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
    except Exception as e:  # no daemon, no access, or no image
        pytest.skip(f"needs Docker and the image {IMAGE}: {type(e).__name__}")
    return client


class Upstream:
    """A local stand-in for the provider: records each request and answers 200."""

    def __init__(self, host="127.0.0.1"):
        self.requests = []
        up = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                up.requests.append((self.path, dict(self.headers), body))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _proxy(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("TSG_NETWORK_TEST_KEY", HOST_KEY)
    raw = {"key": "${{ secrets.TSG_NETWORK_TEST_KEY }}", "models": ["gpt-test"]}
    config = mp.parse_config({"providers": {"openai": raw}})
    origin = f"http://127.0.0.1:{upstream.port}"
    providers = {"openai": dataclasses.replace(mp.PROVIDERS["openai"], origin=origin)}
    return mp.ModelProxy(config, tmp_path / "proxy.jsonl", providers)


def _bridge_listener(client):
    """A listener on the gateway of Docker's default bridge, or None if it cannot bind there."""
    gateway = client.networks.get("bridge").attrs["IPAM"]["Config"][0]["Gateway"]
    try:
        return Upstream(host=gateway)
    except OSError:  # not in the host's network namespace
        return None


def _probe(client, proxy, network, targets):
    container = client.containers.create(
        IMAGE,
        entrypoint=["sleep", "600"],
        network_mode=network,
        volumes=proxy.volumes,
        environment=proxy.container_env(),
        labels=LABEL,
        mem_limit="1g",
        detach=True,
    )
    try:
        container.start()
        proxy.start_relay(container)
        cmd = ["python3", "-I", "-c", PROBE, json.dumps([targets, network == "none"])]
        exit_code, output = container.exec_run(cmd, user="nobody")
        assert exit_code == 0, output.decode()
        container.reload()
        return json.loads(output.decode().strip().splitlines()[-1]), container.attrs
    finally:
        container.remove(force=True)


def test_agent_containers_reach_the_model_proxy_and_nothing_else(client, tmp_path, monkeypatch):
    network = inspect.signature(run_in_container).parameters["network"].default
    assert network == "none"
    upstream = Upstream()
    listener = _bridge_listener(client)
    targets = [] if listener is None else [listener.server.server_address]
    try:
        with _proxy(tmp_path, monkeypatch, upstream) as proxy:
            seen, attrs = _probe(client, proxy, network, targets)
    finally:
        upstream.server.shutdown()
        if listener is not None:
            listener.server.shutdown()
    # no route out: only the loopback device, and nothing but loopback is reachable
    assert seen["ipv4_routes"] == [] and seen["ipv6_devices"] == ["lo"]
    assert seen["test_net"] == "ENETUNREACH"
    assert seen["test_net6"] in ("ENETUNREACH", "EADDRNOTAVAIL")  # no route, or no address
    assert seen["targets"] == ["ENETUNREACH"] * len(targets)  # the host, which an internal
    # Docker network would have let it reach at its gateway address
    # the proxy, through the relay and the mounted socket, forwards only the allowed model
    assert (seen["allowed"], seen["other_model"], seen["no_token"]) == (200, 403, 403)
    assert len(upstream.requests) == 1
    path, headers, _ = upstream.requests[0]
    assert path == "/v1/chat/completions" and headers["Authorization"] == f"Bearer {HOST_KEY}"
    # the container never had the key
    assert not any(HOST_KEY in var for var in attrs["Config"]["Env"])
    assert "OPENAI_API_KEY=" + mp.DUMMY_KEY in attrs["Config"]["Env"]
    log = (tmp_path / "proxy.jsonl").read_text()
    assert HOST_KEY not in log and proxy._token not in log


def test_the_bridge_listener_is_reachable_from_the_default_network(client, tmp_path, monkeypatch):
    """Positive control for the check above: the probe does see a reachable listener."""
    listener = _bridge_listener(client)
    if listener is None:
        pytest.skip("run on the Docker host to bind to the bridge's gateway")
    upstream = Upstream()
    try:
        with _proxy(tmp_path, monkeypatch, upstream) as proxy:
            seen, _ = _probe(client, proxy, "bridge", [listener.server.server_address])
    finally:
        upstream.server.shutdown()
        listener.server.shutdown()
    assert seen["targets"] == ["connected"]
    assert seen["allowed"] == 200  # the proxy works on any network
