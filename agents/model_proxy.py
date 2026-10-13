"""Host-side model proxy: agents call model APIs through it and never hold the API keys.

`agents/run.py` starts one proxy per run: a unix socket in a new directory, which is mounted
read-only into the agent's container at CONTAINER_DIR, and a fresh random token. By default the
container has no network. A relay in it (`model_proxy_relay.py`, started by `run.py`) listens on
127.0.0.1:RELAY_PORT and passes each connection to the socket. The agent gets base URLs and
dummy keys for the providers in its config, for example

    OPENAI_BASE_URL=http://127.0.0.1:18080/t/<token>/openai/v1
    OPENAI_API_KEY=tsgym-model-proxy-dummy-key

and the proxy forwards only

    POST /t/<token>/<provider><path>    <path> in PROVIDERS[<provider>].paths, and the JSON
                                        body's "model" in the agent's list for that provider

to the provider's fixed origin, with the host's API key for that provider. Every other method,
path, provider, model or token is refused; `GET /health` answers 200 and nothing else. The
agent's own auth headers and cookies are dropped, and a provider's reply to a rejected key is
replaced by a generic one, as it can quote part of the key. Forwarding stops after
`max_requests` requests or `max_seconds`. The log (JSON lines in the run directory) records
counts, status codes, sizes and the allowed model ids; never request or response content, the
token or the keys.

Connections are bounded as in the Bedrock proxy of TimeSeriesGym's RL environments
(`rl_envs/tsgym_rl/bedrock_proxy.py`): a socket read or write that waits READ_TIMEOUT_S ends the
connection, the request line, headers and body must arrive within REQUEST_DEADLINE_S (an idle
keep-alive connection ends then too), at most MAX_CONNECTIONS are open at once (beyond that: 503
and close), and the request bodies held in memory are capped in total (MAX_BUFFERED_BYTES;
beyond that: 503). Streaming a response back is not time-limited.
"""

from __future__ import annotations

import hmac
import http.client
import json
import os
import secrets
import shutil
import socket
import socketserver
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from agents.utils import get_env_var

RELAY_PORT = 18080  # on 127.0.0.1 in the agent's container
CONTAINER_DIR = "/run/tsgym-model-proxy"  # the proxy's directory in the container (read-only)
SOCKET_NAME = "proxy.sock"
RELAY_NAME = "relay.py"
RELAY_SOURCE = Path(__file__).with_name("model_proxy_relay.py")
RELAY_WAIT_S = 60  # for the relay to start in the container
LOG_NAME = "model_proxy.jsonl"  # in the run directory
DUMMY_KEY = "tsgym-model-proxy-dummy-key"  # what the agent's SDKs send; the proxy drops it

MAX_BODY = 32 * 1024**2
MAX_BUFFERED_BYTES = 256 * 1024**2  # request bodies held at once, over all connections
READ_TIMEOUT_S = 60.0  # one socket read or write from or to the agent
REQUEST_DEADLINE_S = 120.0  # request line, headers and body, from the first byte waited for
UPSTREAM_TIMEOUT_S = 900.0  # one socket read or write from or to the provider
MAX_CONNECTIONS = 64
MAX_REQUESTS = 2000  # per run, unless the agent's config sets `max_requests`
MAX_SECONDS = 24 * 3600  # per run, unless the agent's config sets `max_seconds`

# response headers passed back to the agent's SDK (its retry logic reads the last four)
RESPONSE_HEADERS = (
    "Content-Type",
    "request-id",
    "x-request-id",
    "Retry-After",
    "retry-after-ms",
    "x-should-retry",
)


@dataclass(frozen=True)
class Provider:
    """A model API: where it is, what may be called, and how the key and base URL are passed."""

    origin: str  # scheme://host[:port]
    paths: frozenset[str]  # request paths that may be forwarded
    key_header: str
    key_prefix: str = ""
    forward_headers: tuple[str, ...] = ()  # request headers passed on besides Content-Type, Accept
    base_path: str = ""  # appended to <proxy>/<provider> for the base-URL variables
    base_url_vars: tuple[str, ...] = ()  # environment variables the agent's SDKs read
    api_key_vars: tuple[str, ...] = ()


PROVIDERS = {
    "openai": Provider(
        origin="https://api.openai.com",
        paths=frozenset({"/v1/chat/completions", "/v1/completions", "/v1/responses"}),
        key_header="Authorization",
        key_prefix="Bearer ",
        base_path="/v1",
        # openai>=1 and litellm read OPENAI_BASE_URL, openai<1 (MLAgentBench) OPENAI_API_BASE
        base_url_vars=("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        api_key_vars=("OPENAI_API_KEY",),
    ),
    "anthropic": Provider(
        origin="https://api.anthropic.com",
        paths=frozenset({"/v1/messages", "/v1/messages/count_tokens", "/v1/complete"}),
        key_header="x-api-key",
        forward_headers=("anthropic-version", "anthropic-beta"),
        base_url_vars=("ANTHROPIC_BASE_URL",),
        api_key_vars=("ANTHROPIC_API_KEY",),
    ),
    "openrouter": Provider(  # OpenAI-compatible
        origin="https://openrouter.ai",
        paths=frozenset({"/api/v1/chat/completions", "/api/v1/completions"}),
        key_header="Authorization",
        key_prefix="Bearer ",
        base_path="/api/v1",
        base_url_vars=("OPENROUTER_BASE_URL", "OPENROUTER_API_BASE"),
        api_key_vars=("OPENROUTER_API_KEY",),
    ),
    "gemini": Provider(  # Gemini's OpenAI-compatible API
        origin="https://generativelanguage.googleapis.com",
        paths=frozenset({"/v1beta/openai/chat/completions"}),
        key_header="Authorization",
        key_prefix="Bearer ",
        base_path="/v1beta/openai",
        base_url_vars=("GEMINI_BASE_URL",),
        api_key_vars=("GEMINI_API_KEY",),
    ),
}


def parse_config(raw: dict, providers: dict[str, Provider] = PROVIDERS) -> dict:
    """Checks an agent's `model_proxy` config and reads each provider's key from the environment.

        model_proxy:
          providers:
            openai:
              key: ${{ secrets.OPENAI_API_KEY }}  # read on the host; never enters the container
              models: [gpt-4.1-2025-04-14]       # the "model" values the agent may send
          max_requests: 1000                     # optional, per run
          max_seconds: 16200                     # optional, per run

    A key must be a `${{ secrets.NAME }}` reference, so that no key is written in a config file.
    """
    if not isinstance(raw, dict) or set(raw) - {"providers", "max_requests", "max_seconds"}:
        raise ValueError("model_proxy takes `providers`, `max_requests` and `max_seconds`")
    entries = raw.get("providers")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("model_proxy.providers must name at least one provider")
    keys, models = {}, {}
    for name, entry in entries.items():
        if name not in providers:
            raise ValueError(f"model_proxy: unknown provider {name!r} (known: {sorted(providers)})")
        if not isinstance(entry, dict) or set(entry) != {"key", "models"}:
            raise ValueError(f"model_proxy.providers.{name} takes exactly `key` and `models`")
        env_var = get_env_var(entry["key"])
        if env_var is None:
            raise ValueError(
                f"model_proxy.providers.{name}.key must be " + "`${{ secrets.ENV_VAR }}`"
            )
        if not os.getenv(env_var):
            raise ValueError(f"Environment variable `{env_var}` is not set!")
        names = entry["models"]
        if not isinstance(names, list) or not names:
            raise ValueError(f"model_proxy.providers.{name}.models must be a non-empty list")
        if not all(isinstance(m, str) and m for m in names):
            raise ValueError(f"model_proxy.providers.{name}.models must hold model ids")
        keys[name], models[name] = os.getenv(env_var), frozenset(names)
    limits = {}
    for field, default in (("max_requests", MAX_REQUESTS), ("max_seconds", MAX_SECONDS)):
        value = raw.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"model_proxy.{field} must be a positive number")
        limits[field] = value
    return {"keys": keys, "models": models, **limits}


def make_handler(cfg: dict, record):
    """The request handler class, and the counters it shares with its proxy."""
    token = cfg["token"].encode()
    providers, keys, models = cfg["providers"], cfg["keys"], cfg["models"]
    deadline_s, max_buffered = cfg["request_deadline"], cfg["max_buffered"]
    lock = threading.Lock()
    counters = {
        "forwarded": 0,
        "refused": 0,
        "upstream_errors": 0,
        "buffered": 0,
        "started": time.monotonic(),
        "closed": False,
    }

    def take_budget() -> tuple[int, str | None]:
        """Count one forwarded request: (its number, None), or (0, why) once a budget is spent."""
        with lock:
            if counters["forwarded"] >= cfg["max_requests"]:
                return 0, "request budget exhausted"
            if time.monotonic() - counters["started"] > cfg["max_seconds"]:
                return 0, "time budget exhausted"
            counters["forwarded"] += 1
            return counters["forwarded"], None

    def reserve(n: int) -> bool:
        with lock:
            if counters["buffered"] + n > max_buffered:
                return False
            counters["buffered"] += n
            return True

    def release(n: int) -> None:
        with lock:
            counters["buffered"] -= n

    def count(name: str) -> None:
        with lock:
            counters[name] += 1

    def route(raw_path: str) -> tuple[str, str] | None:
        """(provider, path) of an allowed /t/<token>/<provider><path>, else None."""
        path = urllib.parse.urlsplit(raw_path).path
        if not path.startswith("/t/"):
            return None
        given, _, rest = path[3:].partition("/")
        if not hmac.compare_digest(given.encode("latin-1"), token):
            return None
        name, _, sub = rest.partition("/")
        if name not in keys or f"/{sub}" not in providers[name].paths:
            return None
        return name, f"/{sub}"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = cfg["read_timeout"]  # socket timeout of the connection

        def log_message(self, *args):  # silence the default access log: paths hold the token
            return

        def handle_one_request(self):
            # The whole request must arrive by the deadline (a socket timeout alone lets a
            # client that sends a byte now and then keep the connection forever).
            self._deadline = threading.Timer(deadline_s, self._abort)
            self._deadline.daemon = True
            self._deadline.start()
            try:
                super().handle_one_request()
            finally:
                self._deadline.cancel()

        def _abort(self) -> None:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        def _send_json(self, code: int, message: str, close: bool = False) -> None:
            body = json.dumps({"error": {"message": message}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _refuse(self, code: int, why: str, detail: str = "") -> None:
            """Refuses the request; the log gets `why` only (`detail` is the agent's own data)."""
            count("refused")
            record(event="refused", code=code, why=why)
            self.close_connection = True  # an unread body must not be parsed as a request
            self._send_json(code, f"model proxy: {why}{detail}", close=True)

        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            if urllib.parse.urlsplit(self.path).path == "/health":
                return self._send_json(200, "ok")
            self._refuse(405, "method not allowed")

        def do_PUT(self):  # noqa: N802
            self._refuse(405, "method not allowed")

        do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_PUT  # noqa: N815

        def do_POST(self):  # noqa: N802
            allowed = route(self.path)
            if allowed is None:
                return self._refuse(403, "path not allowed")
            if counters["closed"]:
                return self._refuse(503, "the run has ended")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._refuse(400, "bad Content-Length")
            if length <= 0 or length > MAX_BODY:
                return self._refuse(413, "bad body size")
            if not reserve(length):
                return self._refuse(503, "too many request bytes in flight")
            try:
                self._forward(*allowed, length)
            finally:
                release(length)

        def _forward(self, name: str, path: str, length: int) -> None:
            try:
                body = self.rfile.read(length)
            except OSError:  # timed out or cut off
                body = b""
            if len(body) != length:
                return self._refuse(400, "incomplete body")
            self._deadline.cancel()  # the request is in; the response may stream for long
            try:
                model = json.loads(body).get("model")
            except (ValueError, AttributeError):
                return self._refuse(400, "body must be a JSON object")
            if not isinstance(model, str) or model not in models[name]:
                return self._refuse(403, "model not allowed", f": {str(model)[:80]}")
            n, exhausted = take_budget()
            if exhausted:  # refused outside the lock: _refuse takes it again
                return self._refuse(429, exhausted)
            provider = providers[name]
            headers = {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Accept": self.headers.get("Accept", "application/json"),
            }
            for header in provider.forward_headers:
                if self.headers.get(header) is not None:
                    headers[header] = self.headers[header]
            headers[provider.key_header] = provider.key_prefix + keys[name]
            origin = urllib.parse.urlsplit(provider.origin)
            connection = http.client.HTTPSConnection
            if origin.scheme == "http":  # the tests' local stand-ins
                connection = http.client.HTTPConnection
            conn = connection(origin.hostname, origin.port, timeout=UPSTREAM_TIMEOUT_S)
            started, status, sent, complete = time.monotonic(), None, 0, False
            try:
                try:
                    conn.request("POST", path, body=body, headers=headers)
                    resp = conn.getresponse()
                except (OSError, http.client.HTTPException) as e:
                    count("upstream_errors")
                    record(event="upstream_error", n=n, provider=name, error=type(e).__name__)
                    self.close_connection = True
                    return self._send_json(502, "model proxy: the provider could not be reached")
                status = resp.status
                if status in (401, 403):  # the provider's message can quote part of the key
                    return self._send_json(status, "model proxy: the provider refused its key")
                self.send_response(status)
                for header in RESPONSE_HEADERS:
                    if resp.getheader(header):
                        self.send_header(header, resp.getheader(header))
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                sent, complete = self._stream(resp)
            finally:
                conn.close()
                if status is not None:
                    record(
                        event="forwarded",
                        n=n,
                        provider=name,
                        path=path,
                        model=model,
                        status=status,
                        req_bytes=length,
                        resp_bytes=sent,
                        seconds=round(time.monotonic() - started, 3),
                    )
            if complete:  # end the response after its log entry: the run may end right then
                try:
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    self.close_connection = True

        def _stream(self, resp: http.client.HTTPResponse) -> tuple[int, bool]:
            """Passes the provider's response body on in chunks: (bytes sent, whether all)."""
            sent = 0
            try:
                while chunk := resp.read1(65536):
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                    sent += len(chunk)
            except (OSError, http.client.HTTPException):  # the provider or the agent went away
                self.close_connection = True
                return sent, False
            return sent, True

    return Handler, counters


class CappedUnixServer(socketserver.ThreadingUnixStreamServer):
    """One thread per connection, at most `max_connections` open at once. A connection beyond
    that gets 503 and is closed at once."""

    daemon_threads = True

    def __init__(self, path: str, handler, max_connections: int = MAX_CONNECTIONS):
        super().__init__(path, handler)
        self.max_connections = max_connections
        self._open = 0
        self._open_lock = threading.Lock()

    def _take(self) -> bool:
        with self._open_lock:
            if self._open >= self.max_connections:
                return False
            self._open += 1
            return True

    def _give(self) -> None:
        with self._open_lock:
            self._open -= 1

    def process_request(self, request, client_address):
        if not self._take():
            try:
                request.settimeout(5)
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n"
                    b"Connection: close\r\n\r\n"
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._give()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._give()

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], OSError):  # the agent went away, or a deadline passed
            return
        super().handle_error(request, client_address)


class ModelProxy:
    """The model proxy of one run (see the module docstring); a context manager.

    `config` is the output of `parse_config`. `providers` and the keyword limits exist for tests.
    """

    relay_command = (
        "python3",
        "-I",
        f"{CONTAINER_DIR}/{RELAY_NAME}",
        str(RELAY_PORT),
        f"{CONTAINER_DIR}/{SOCKET_NAME}",
    )
    relay_wait_command = (*relay_command[:3], "--wait", str(RELAY_PORT), str(RELAY_WAIT_S))

    def __init__(
        self,
        config: dict,
        log_path: Path,
        providers: dict[str, Provider] = PROVIDERS,
        *,
        read_timeout: float = READ_TIMEOUT_S,
        request_deadline: float = REQUEST_DEADLINE_S,
        max_connections: int = MAX_CONNECTIONS,
        max_buffered: int = MAX_BUFFERED_BYTES,
    ):
        self._token = secrets.token_hex(24)  # never reused: each run gets its own
        self._names = sorted(config["keys"])
        self._providers = providers
        self._closed = False
        self._log_lock = threading.Lock()
        self._log = open(log_path, "a", buffering=1)
        # a short path (unix socket paths are limited to about 100 bytes)
        self.dir = Path(tempfile.mkdtemp(prefix="tsgym-model-proxy-"))
        self.socket_path = self.dir / SOCKET_NAME
        try:
            # The container's relay must reach the socket also when the runtime maps user ids
            # (sysbox), so the socket is world-writable. The token guards it, not the file mode.
            self.dir.chmod(0o711)
            shutil.copyfile(RELAY_SOURCE, self.dir / RELAY_NAME)
            (self.dir / RELAY_NAME).chmod(0o644)
            cfg = {
                **config,
                "token": self._token,
                "providers": providers,
                "read_timeout": read_timeout,
                "request_deadline": request_deadline,
                "max_buffered": max_buffered,
            }
            handler, self.counters = make_handler(cfg, self._record)
            self._server = CappedUnixServer(str(self.socket_path), handler, max_connections)
            self.socket_path.chmod(0o666)
        except BaseException:
            shutil.rmtree(self.dir, ignore_errors=True)
            self._log.close()
            raise
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self._record(event="started", providers=self._names)

    def _record(self, **fields) -> None:
        with self._log_lock:
            if not self._log.closed:  # a response may still be streaming at close
                self._log.write(json.dumps({"t": round(time.time(), 3), **fields}) + "\n")

    def container_env(self) -> dict[str, str]:
        """Environment variables for the agent's container: base URLs and dummy keys."""
        base = f"http://127.0.0.1:{RELAY_PORT}/t/{self._token}"
        env = {"MODEL_PROXY_URL": base}
        for name in self._names:
            provider = self._providers[name]
            for var in provider.base_url_vars:
                env[var] = f"{base}/{name}{provider.base_path}"
            for var in provider.api_key_vars:
                env[var] = DUMMY_KEY
        return env

    @property
    def volumes(self) -> dict:
        return {str(self.dir): {"bind": CONTAINER_DIR, "mode": "ro"}}

    @classmethod
    def for_agent(cls, agent, run_dir: Path) -> ModelProxy | None:
        """The proxy of a run of `agent`, or None if its config has no `model_proxy`."""
        config = getattr(agent, "model_proxy", None)
        return None if config is None else cls(config, run_dir / LOG_NAME)

    def start_relay(self, container) -> None:
        """Starts the relay in the (started) container and waits until the proxy answers
        through it. The relay runs as `nobody`, so the agent cannot stop it."""
        container.exec_run(list(self.relay_command), user="nobody", detach=True)
        exit_code, _ = container.exec_run(list(self.relay_wait_command), user="nobody")
        if exit_code != 0:
            raise RuntimeError(
                f"The model proxy's relay did not answer within {RELAY_WAIT_S} seconds. The "
                "agent's image must have `python3`; check the logs."
            )

    def summary(self) -> str:
        c = self.counters
        return (
            f"Model proxy: {c['forwarded']} requests forwarded, {c['refused']} refused, "
            f"{c['upstream_errors']} provider errors."
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.counters["closed"] = True  # requests on connections still open are refused
        self._server.shutdown()
        self._server.server_close()
        shutil.rmtree(self.dir, ignore_errors=True)
        self._record(event="closed", **{k: self.counters[k] for k in ("forwarded", "refused")})
        with self._log_lock:
            self._log.close()

    def __enter__(self) -> ModelProxy:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
