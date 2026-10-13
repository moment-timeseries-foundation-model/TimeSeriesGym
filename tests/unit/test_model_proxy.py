"""The host-side model proxy and its in-container relay.

The proxy forwards to local stand-ins for the providers; no test calls a real provider.
"""

import dataclasses
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agents import model_proxy as mp

REPO = Path(__file__).parents[2]
HOST_KEYS = {
    "openai": "sk-host-openai-0123456789",
    "anthropic": "sk-ant-host-0123456789",
    "openrouter": "sk-or-host-0123456789",
    "gemini": "gm-host-0123456789",
}
# what an SDK appends to its base URL, and an allowed model, per provider
CALLS = {
    "openai": ("OPENAI_BASE_URL", "/chat/completions", "gpt-test"),
    "anthropic": ("ANTHROPIC_BASE_URL", "/v1/messages", "claude-test"),
    "openrouter": ("OPENROUTER_BASE_URL", "/chat/completions", "vendor/llama-test"),
    "gemini": ("GEMINI_BASE_URL", "/chat/completions", "gemini-test"),
}


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=10):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_path)


class Upstream:
    """A local stand-in for the providers: records each request, then answers `status` with
    `chunks` (sent `delay` s apart)."""

    def __init__(self, status=200, chunks=(b'{"ok": true}',), delay=0.0):
        self.requests = []
        up = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                return

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                up.requests.append((self.path, dict(self.headers), body))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-request-id", "req-1")
                self.send_header("Content-Length", str(sum(map(len, chunks))))
                self.end_headers()
                for chunk in chunks:
                    time.sleep(delay)
                    self.wfile.write(chunk)
                    self.wfile.flush()

        class Server(ThreadingHTTPServer):
            def handle_error(self, request, client_address):  # the proxy hung up early
                pass

        self.server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.providers = {
            name: dataclasses.replace(p, origin=f"http://127.0.0.1:{self.server.server_port}")
            for name, p in mp.PROVIDERS.items()
        }


def _config(monkeypatch, providers=("openai",), **limits):
    raw = {"providers": {}, **limits}
    for name in providers:
        monkeypatch.setenv(f"TEST_{name.upper()}_KEY", HOST_KEYS[name])
        raw["providers"][name] = {
            "key": "${{ secrets.TEST_" + name.upper() + "_KEY }}",
            "models": [CALLS[name][2]],
        }
    return mp.parse_config(raw)


@pytest.fixture
def make_proxy(tmp_path, monkeypatch):
    made = []

    def make(upstream=None, providers=("openai",), limits=None, **caps):
        upstream = upstream or Upstream()
        config = _config(monkeypatch, providers, **(limits or {}))
        proxy = mp.ModelProxy(config, tmp_path / "proxy.jsonl", upstream.providers, **caps)
        made.append((proxy, upstream))
        return proxy, upstream

    yield make
    for proxy, upstream in made:
        proxy.close()
        upstream.server.shutdown()


def _post(proxy, path, body=None, headers=None, model="gpt-test"):
    body = json.dumps({"model": model, "messages": []}).encode() if body is None else body
    conn = UnixHTTPConnection(proxy.socket_path)
    conn.request(
        "POST", path, body=body, headers={"Content-Type": "application/json", **(headers or {})}
    )
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


def _url_path(proxy, provider):
    var, suffix, _ = CALLS[provider]
    return proxy.container_env()[var].removeprefix(f"http://127.0.0.1:{mp.RELAY_PORT}") + suffix


# ---------------------------------------------------------------- what is forwarded


@pytest.mark.parametrize("provider", sorted(CALLS))
def test_proxy_forwards_allowed_calls_with_the_hosts_key(make_proxy, provider):
    proxy, up = make_proxy(providers=(provider,))
    agent_headers = {
        "Authorization": "Bearer tsgym-model-proxy-dummy-key",
        "x-api-key": "dummy",
        "Cookie": "c=1",
        "anthropic-version": "2023-06-01",
    }
    resp, data = _post(
        proxy, _url_path(proxy, provider), headers=agent_headers, model=CALLS[provider][2]
    )
    assert resp.status == 200 and data == b'{"ok": true}'
    assert resp.getheader("x-request-id") == "req-1"
    path, sent, body = up.requests[0]
    assert json.loads(body)["model"] == CALLS[provider][2]
    p = mp.PROVIDERS[provider]
    assert path in p.paths  # the provider's own path, without the token
    sent = {k.lower(): v for k, v in sent.items()}
    assert sent[p.key_header.lower()] == p.key_prefix + HOST_KEYS[provider]
    assert "cookie" not in sent
    if p.key_header.lower() != "authorization":
        assert "authorization" not in sent
    if p.key_header.lower() != "x-api-key":
        assert "x-api-key" not in sent
    assert ("anthropic-version" in sent) is (provider == "anthropic")
    assert proxy.counters["forwarded"] == 1


def test_proxy_streams_the_response_past_the_request_deadline(make_proxy):
    """The deadline bounds receiving the request, not the (long) response."""
    up = Upstream(chunks=(b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"), delay=0.6)
    proxy, _ = make_proxy(up, read_timeout=1.0, request_deadline=1.0)
    resp, data = _post(proxy, _url_path(proxy, "openai"))
    assert resp.status == 200 and data == b"data: 1\n\ndata: 2\n\ndata: [DONE]\n\n"
    deadline = time.monotonic() + 5  # released by the handler thread after the response
    while proxy.counters["buffered"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert proxy.counters["buffered"] == 0


def test_proxy_keeps_connections_alive_between_calls(make_proxy):
    proxy, up = make_proxy()
    conn = UnixHTTPConnection(proxy.socket_path)
    for _ in range(3):  # what SDK clients do
        conn.request("POST", _url_path(proxy, "openai"), body=b'{"model": "gpt-test"}')
        resp = conn.getresponse()
        assert resp.status == 200 and resp.read() == b'{"ok": true}'
    conn.close()
    assert len(up.requests) == 3


# ---------------------------------------------------------------- what is refused


@pytest.mark.parametrize(
    "path, model, code",
    [
        ("/t/wrong/openai/v1/chat/completions", "gpt-test", 403),
        ("/openai/v1/chat/completions", "gpt-test", 403),
        ("/t/{token}/anthropic/v1/messages", "gpt-test", 403),  # not in the agent's config
        ("/t/{token}/azure/v1/chat/completions", "gpt-test", 403),
        ("/t/{token}/openai/v1/embeddings", "gpt-test", 403),
        ("/t/{token}/openai/v1/files", "gpt-test", 403),
        ("/t/{token}/openai/v1/chat/completions/../../files", "gpt-test", 403),
        ("/t/{token}/openai/v1/chat%2Fcompletions", "gpt-test", 403),
        ("/t/{token}/openai/v1/chat/completions", "gpt-other", 403),
        ("/t/{token}/openai/v1/chat/completions", ["gpt-test"], 403),
        ("/t/{token}/openai/v1/chat/completions", None, 403),
    ],
)
def test_proxy_refuses_anything_but_an_allowed_model_call(make_proxy, path, model, code):
    proxy, up = make_proxy()
    resp, data = _post(proxy, path.format(token=proxy._token), model=model)
    assert resp.status == code
    assert up.requests == [] and proxy.counters["forwarded"] == 0
    assert proxy._token not in data.decode()


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH"])
def test_proxy_refuses_other_methods(make_proxy, method):
    proxy, up = make_proxy()
    conn = UnixHTTPConnection(proxy.socket_path)
    conn.request(method, _url_path(proxy, "openai"), body=b'{"model": "gpt-test"}')
    assert conn.getresponse().status == 405 and up.requests == []
    conn.close()


def test_proxy_health_check_needs_no_token(make_proxy):
    proxy, _ = make_proxy()
    conn = UnixHTTPConnection(proxy.socket_path)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    assert resp.status == 200 and proxy._token not in resp.read().decode()
    conn.close()


@pytest.mark.parametrize("body, code", [(b"not json", 400), (b"[1, 2]", 400), (b'"gpt-test"', 400)])
def test_proxy_refuses_bodies_that_are_not_json_objects(make_proxy, body, code):
    proxy, up = make_proxy()
    resp, _ = _post(proxy, _url_path(proxy, "openai"), body=body)
    assert resp.status == code and up.requests == []


def test_proxy_stops_forwarding_once_a_budget_is_spent(make_proxy):
    proxy, up = make_proxy(limits={"max_requests": 2})
    codes = [_post(proxy, _url_path(proxy, "openai"))[0].status for _ in range(4)]
    assert codes == [200, 200, 429, 429]  # and no deadlock on the budget's lock
    assert len(up.requests) == 2 and proxy.counters["refused"] == 2


def test_proxy_stops_forwarding_after_its_time_budget(make_proxy):
    proxy, up = make_proxy(limits={"max_seconds": 0.5})
    assert _post(proxy, _url_path(proxy, "openai"))[0].status == 200
    time.sleep(0.7)
    assert _post(proxy, _url_path(proxy, "openai"))[0].status == 429
    assert len(up.requests) == 1


def test_proxy_hides_the_providers_reply_to_a_rejected_key(make_proxy):
    """Providers quote part of a wrong key in their error message."""
    up = Upstream(status=401, chunks=(b'{"error": "Incorrect API key: sk-host-openai-...6789"}',))
    proxy, _ = make_proxy(up)
    resp, data = _post(proxy, _url_path(proxy, "openai"))
    assert resp.status == 401 and b"6789" not in data and b"sk-" not in data


def test_proxy_passes_other_provider_errors_on(make_proxy):
    up = Upstream(status=429, chunks=(b'{"error": "rate limited"}',))
    proxy, _ = make_proxy(up)
    resp, data = _post(proxy, _url_path(proxy, "openai"))
    assert resp.status == 429 and data == b'{"error": "rate limited"}'


def test_proxy_answers_502_when_the_provider_is_unreachable(tmp_path, monkeypatch):
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    providers = {
        "openai": dataclasses.replace(mp.PROVIDERS["openai"], origin=f"http://127.0.0.1:{port}")
    }
    with mp.ModelProxy(_config(monkeypatch), tmp_path / "log.jsonl", providers) as proxy:
        resp, _ = _post(proxy, _url_path(proxy, "openai"))
        assert resp.status == 502 and proxy.counters["upstream_errors"] == 1


# ---------------------------------------------------------------- bounded connections


def _closed_within(sock: socket.socket, seconds: float) -> bool:
    """Whether the server closes the connection (EOF) within `seconds`."""
    sock.settimeout(seconds)
    try:
        while sock.recv(4096):
            pass
        return True
    except TimeoutError:
        return False
    except OSError:
        return True


def _connect(proxy) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(proxy.socket_path))
    return sock


def test_proxy_closes_idle_connections(make_proxy):
    proxy, _ = make_proxy(read_timeout=0.5, request_deadline=5)
    sock = _connect(proxy)
    assert _closed_within(sock, 4)
    sock.close()


def test_proxy_closes_a_request_that_trickles_in(make_proxy):
    """A byte every 0.2 s never trips a 0.5 s socket timeout; the request deadline ends it."""
    proxy, _ = make_proxy(read_timeout=0.5, request_deadline=1.0)
    sock = _connect(proxy)
    t0, closed = time.monotonic(), False
    for byte in b"POST /t/x/openai/v1/chat/completions HTTP/1.1\r\nX-Slow: " + b"a" * 100:
        try:
            sock.sendall(bytes([byte]))
        except OSError:
            closed = True
            break
        time.sleep(0.2)
        if time.monotonic() - t0 > 4:
            break
    assert closed or _closed_within(sock, 1)
    assert time.monotonic() - t0 < 4
    sock.close()


def test_proxy_caps_open_connections(make_proxy):
    proxy, _ = make_proxy(max_connections=2)
    held = [_connect(proxy) for _ in range(2)]
    time.sleep(0.2)
    extra = _connect(proxy)
    extra.settimeout(5)
    assert extra.recv(100).startswith(b"HTTP/1.1 503")
    for s in [*held, extra]:
        s.close()
    deadline, status = time.monotonic() + 5, None
    while status != 200 and time.monotonic() < deadline:  # the slots are free again
        conn = UnixHTTPConnection(proxy.socket_path)
        conn.request("GET", "/health")
        status = conn.getresponse().status
        conn.close()
        time.sleep(0.05)
    assert status == 200


@pytest.mark.parametrize(
    "headers, body, code",
    [
        ({"Content-Length": "abc"}, b"", 400),
        ({"Content-Length": str(64 << 20)}, b"", 413),
        ({"Content-Length": "0"}, b"", 413),
        ({"Content-Length": "100"}, b"{}", 400),  # cut short: the deadline ends the read
    ],
)
def test_proxy_refuses_bad_bodies(make_proxy, headers, body, code):
    proxy, _ = make_proxy(read_timeout=0.5, request_deadline=5)
    sock = _connect(proxy)
    head = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    path = _url_path(proxy, "openai")
    sock.sendall(f"POST {path} HTTP/1.1\r\n{head}\r\n".encode() + body)
    sock.settimeout(5)
    assert sock.recv(200).startswith(f"HTTP/1.1 {code}".encode())
    sock.close()


def test_proxy_bounds_the_request_bytes_it_holds(make_proxy):
    proxy, up = make_proxy(max_buffered=1000)
    resp, _ = _post(
        proxy,
        _url_path(proxy, "openai"),
        body=b'{"model": "gpt-test", "x": "' + b"x" * 2000 + b'"}',
    )
    assert resp.status == 503 and up.requests == []
    deadline = time.monotonic() + 5
    while proxy.counters["buffered"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert proxy.counters["buffered"] == 0


# ---------------------------------------------------------------- what the agent and the log see


def test_the_container_gets_base_urls_and_dummy_keys_only(make_proxy):
    proxy, _ = make_proxy(providers=("openai", "anthropic"))
    env = proxy.container_env()
    base = f"http://127.0.0.1:{mp.RELAY_PORT}/t/{proxy._token}"
    assert env["OPENAI_BASE_URL"] == env["OPENAI_API_BASE"] == f"{base}/openai/v1"
    assert env["ANTHROPIC_BASE_URL"] == f"{base}/anthropic"
    assert env["OPENAI_API_KEY"] == env["ANTHROPIC_API_KEY"] == mp.DUMMY_KEY
    assert env["MODEL_PROXY_URL"] == base
    assert "OPENROUTER_BASE_URL" not in env  # only the agent's providers
    assert not any(key in value for key in HOST_KEYS.values() for value in env.values())
    assert proxy.volumes == {str(proxy.dir): {"bind": mp.CONTAINER_DIR, "mode": "ro"}}
    # the relay command names the files the proxy put in its directory
    script, port, sock = proxy.relay_command[-3:]
    assert Path(script).name in os.listdir(proxy.dir) and Path(sock).name in os.listdir(proxy.dir)
    assert port == str(mp.RELAY_PORT) and Path(script).parent == Path(sock).parent == Path(
        mp.CONTAINER_DIR
    )
    assert proxy.socket_path.stat().st_mode & 0o777 == 0o666  # also behind a user-id mapping


def test_proxy_log_has_no_content_token_or_key(make_proxy, tmp_path):
    proxy, _ = make_proxy(Upstream(chunks=(b'{"answer": "RESPONSE-TEXT"}',)))
    body = json.dumps({"model": "gpt-test", "messages": [{"content": "PROMPT-TEXT"}]}).encode()
    _post(proxy, _url_path(proxy, "openai"), body=body)
    _post(proxy, _url_path(proxy, "openai"), model="SMUGGLED-MODEL")
    _post(proxy, f"/t/{proxy._token}/openai/v1/files")
    token = proxy._token
    proxy.close()
    log = (tmp_path / "proxy.jsonl").read_text()
    events = [json.loads(line)["event"] for line in log.splitlines()]
    assert events == ["started", "forwarded", "refused", "refused", "closed"]
    for secret in (token, HOST_KEYS["openai"], "PROMPT-TEXT", "SMUGGLED-MODEL", "RESPONSE-TEXT"):
        assert secret not in log


def test_proxy_refuses_after_close_and_removes_its_directory(make_proxy):
    proxy, _ = make_proxy()
    conn = UnixHTTPConnection(proxy.socket_path)
    conn.request("GET", "/health")
    conn.getresponse().read()
    proxy.close()
    assert not proxy.dir.exists()
    conn.request("POST", _url_path(proxy, "openai"), body=b'{"model": "gpt-test"}')
    assert conn.getresponse().status == 503  # a keep-alive connection that outlived the run
    conn.close()


def test_the_relay_passes_connections_to_the_proxy(make_proxy):
    proxy, up = make_proxy()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    relay = subprocess.Popen(
        [sys.executable, "-I", str(proxy.dir / mp.RELAY_NAME), str(port), str(proxy.socket_path)]
    )
    try:
        deadline, status = time.monotonic() + 10, None
        while status is None and time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                resp.read()
                status = resp.status
            except OSError:
                time.sleep(0.1)
        assert status == 200
        conn.request("POST", _url_path(proxy, "openai"), body=b'{"model": "gpt-test"}')
        resp = conn.getresponse()
        assert resp.status == 200 and resp.read() == b'{"ok": true}'
        conn.request("POST", _url_path(proxy, "openai"), body=b'{"model": "gpt-other"}')
        assert conn.getresponse().status == 403
        conn.close()
        # what run.py waits with before it starts the agent
        wait = [sys.executable, "-I", str(proxy.dir / mp.RELAY_NAME), "--wait", str(port), "10"]
        assert subprocess.run(wait, timeout=30).returncode == 0
    finally:
        relay.kill()
        relay.wait()
    assert len(up.requests) == 1
    wait[-2:] = [str(port), "1"]  # nothing listens there now
    assert subprocess.run(wait, timeout=30).returncode == 1


# ---------------------------------------------------------------- agent configs


def test_proxy_config_needs_secret_references_and_known_fields(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", HOST_KEYS["openai"])
    good = {"key": "${{ secrets.TEST_OPENAI_KEY }}", "models": ["gpt-test"]}
    parsed = mp.parse_config({"providers": {"openai": good}, "max_requests": 5})
    assert parsed["keys"] == {"openai": HOST_KEYS["openai"]}
    assert parsed["models"] == {"openai": frozenset({"gpt-test"})}
    assert parsed["max_requests"] == 5 and parsed["max_seconds"] == mp.MAX_SECONDS
    bad = [
        {"providers": {"openai": {**good, "key": HOST_KEYS["openai"]}}},  # a key in the file
        {"providers": {"openai": {**good, "key": "${{ secrets.TEST_UNSET_KEY }}"}}},
        {"providers": {"azure": good}},
        {"providers": {"openai": {**good, "model": ["x"]}}},
        {"providers": {"openai": {**good, "models": []}}},
        {"providers": {"openai": {**good, "models": "gpt-test"}}},
        {"providers": {}},
        {"providers": {"openai": good}, "max_request": 5},
        {"providers": {"openai": good}, "max_requests": 0},
        {"providers": {"openai": good}, "max_seconds": True},
    ]
    for raw in bad:
        with pytest.raises(ValueError) as e:
            mp.parse_config(raw)
        assert HOST_KEYS["openai"] not in str(e.value)
