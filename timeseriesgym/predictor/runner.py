"""Predictor side of the predictor protocol: import the submitted predict.py and serve the
grader's batches (wire format: protocol.py). The grader starts this file through a launcher
(launch.py); the agent's local check (check.py) starts it the same way.

Usage: python -I runner.py <predictor dir>

It runs with the Python that has the predictor's packages, so it imports only the standard
library and numpy, and nothing from timeseriesgym. The protocol streams are private copies of
fds 0 and 1; fd 1 is pointed at stderr before predict.py runs, so anything the predictor prints
goes to the log, not into the protocol.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
import traceback


def _read_exact(f, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = f.read(n - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


def main(argv: list[str]) -> int:
    predictor_dir = os.path.abspath(argv[0])
    proto_in = os.fdopen(os.dup(0), "rb", buffering=0)
    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull)  # lives as long as the process
    sys.stdout = sys.stderr

    import numpy as np

    def send(header: dict, arrays=()) -> None:
        head = json.dumps(header).encode()
        bufs = [np.ascontiguousarray(a) for a in arrays]
        size = 4 + len(head) + sum(b.nbytes for b in bufs)
        proto_out.write(struct.pack("<QI", size, len(head)) + head)
        for b in bufs:
            proto_out.write(memoryview(b).cast("B"))

    def fail(what: str) -> int:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            send({"type": "error", "error": f"{what}: {tb[-4000:]}"})
        except OSError:
            pass
        return 1

    try:
        os.chdir(predictor_dir)
        sys.path.insert(0, predictor_dir)
        entry = os.path.join(predictor_dir, "predict.py")
        spec = importlib.util.spec_from_file_location("predict", entry)
        module = importlib.util.module_from_spec(spec)
        sys.modules["predict"] = module
        spec.loader.exec_module(module)
        predict = getattr(module, "predict", None)
        if not callable(predict):
            raise TypeError("predict.py defines no function predict(windows)")
    except BaseException:  # reported to the grader
        return fail("importing predict.py")
    send({"type": "ready"})

    while True:
        try:
            (size,) = struct.unpack("<Q", _read_exact(proto_in, 8))
            payload = bytearray(_read_exact(proto_in, size))
        except EOFError:
            return 0
        (hlen,) = struct.unpack_from("<I", payload, 0)
        header = json.loads(bytes(payload[4 : 4 + hlen]))
        if header.get("type") == "end":
            return 0
        offset = 4 + hlen
        windows = []
        for meta in header["windows"]:
            t = meta.pop("target")
            dtype = np.dtype(t["dtype"])
            count = 1
            for n in t["shape"]:
                count *= int(n)
            arr = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
            meta["target"] = arr.reshape(t["shape"])
            offset += count * dtype.itemsize
            windows.append(meta)
        try:
            out = predict(windows)
            if isinstance(out, np.ndarray) and out.ndim >= 1:
                out = list(out)
            if not isinstance(out, list | tuple) or len(out) != len(windows):
                n = len(out) if isinstance(out, list | tuple) else type(out).__name__
                raise TypeError(f"predict() returned {n} forecasts for {len(windows)} windows")
            arrays = []
            for a in out:
                a = np.asarray(a)
                if a.dtype not in (np.float32, np.float64):
                    a = a.astype(np.float64)
                arrays.append(np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False)))
        except BaseException:  # reported to the grader
            return fail(f"predict() on call {header.get('call')}")
        send(
            {
                "type": "forecasts",
                "arrays": [{"dtype": a.dtype.str, "shape": list(a.shape)} for a in arrays],
            },
            arrays,
        )


def serve(work_dir, job: dict) -> int:
    """The same, as `function(work_dir, job)` for a sandbox that runs entry points: serves the
    predictor in work_dir / job["predictor"] on fds 0 and 1 (which must be the pipes to the
    grader, with nothing written to fd 1 before)."""
    return main([os.path.join(str(work_dir), job.get("predictor", "predictor"))])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
