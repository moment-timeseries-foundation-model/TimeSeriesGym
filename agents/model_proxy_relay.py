"""Relay to the model proxy, run in the agent's container (`agents/run.py` starts it).

The container has no network. This relay listens on 127.0.0.1:<port> and passes each
connection, unchanged, to the model proxy's unix socket, which is mounted from the host (see
`agents/model_proxy.py`). It holds no credentials: the proxy checks every request. Standard
library only, as it runs with whatever `python3` the agent's image has.

    python3 -I relay.py <port> <socket>
    python3 -I relay.py --wait <port> <seconds>   # until the proxy answers through the relay
"""

import http.client
import socket
import sys
import threading
import time

MAX_CONNECTIONS = 64  # the proxy caps them too; this bounds the relay's threads


def copy(src, dst):
    """Copies from src to dst until src ends, then ends dst's write side."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    try:
        dst.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def relay(client, path, slots):
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(path)
        to_proxy = threading.Thread(target=copy, args=(client, upstream), daemon=True)
        to_proxy.start()
        copy(upstream, client)
        # the proxy has closed the connection: stop reading from the agent too
        try:
            client.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        to_proxy.join()
    except OSError:
        pass
    finally:
        client.close()
        upstream.close()
        slots.release()


def wait(port, seconds):
    """0 once GET /health through the relay answers 200, 1 if that takes `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return 0
        except (OSError, http.client.HTTPException):
            pass
        finally:
            conn.close()
        time.sleep(0.5)
    return 1


def main(argv):
    if argv[1] == "--wait":
        return wait(int(argv[2]), float(argv[3]))
    port, path = int(argv[1]), argv[2]
    server = socket.create_server(("127.0.0.1", port))
    slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
    while True:
        try:
            client, _ = server.accept()
        except OSError:  # e.g. out of file descriptors: wait for connections to end
            time.sleep(0.1)
            continue
        if not slots.acquire(blocking=False):
            client.close()
            continue
        threading.Thread(target=relay, args=(client, path, slots), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
