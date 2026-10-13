"""Predictor protocol: the grader sends forecast windows to a submitted predictor and collects
its forecasts. The grader (grading.py) runs the predictor through a launcher (launch.py); the
agent's local check (check.py, a copy of this package in the public data) runs it the same way.
Standalone on purpose: stdlib and numpy only.

Wire format, over the predictor process's stdin (requests) and stdout (replies): frames of an
8-byte little-endian length and a payload. A payload is a 4-byte little-endian header length, a
UTF-8 JSON header, then the raw bytes (C order, little-endian) of the arrays the header lists.

    predictor -> grader  {"type": "ready"}                  once predict.py is imported
    grader -> predictor  {"type": "batch", "windows": [...]} + one context array per window
    predictor -> grader  {"type": "forecasts", "arrays": [{"dtype", "shape"}, ...]} + arrays
                         or {"type": "error", "error": "<traceback>"}
    grader -> predictor  {"type": "end"}

Leakage. The predictor is one process that lives for the whole run, so it can remember every
context it was sent. Windows are therefore sent in rounds (schedule()). Every window belongs to
a timeline: by default its own series; series that share a clock and a source share one (TSL's
four ETT datasets: ETTh1 is ETTm1 sampled at the hour; the series of one GIFT-Eval dataset).
Within a timeline, windows go in order of their forecast start time (`clock`), and the windows
with the same start form one tick; round r holds the r-th tick of every timeline. A context ends
where its window's forecast period starts, so every value a round reveals precedes, in time,
the forecast period of every window of its timeline in that round and in later rounds: a test
value reaches the predictor only after the predictor has committed its forecast for it, also
through a related series. This concerns what the calls reveal. The public data follow
GIFT-Eval's per-series split, so a series that ends later can hold values from inside the test
period of another series of its dataset (in kdd_cup_2018_with_missing/H, 60 of the 270 series
end 8 hours after the others). The grader keeps each reply before it sends the next batch, and a
later reply cannot change it. Different timelines are taken to be independent.

Replies are untrusted: any failure to read one (deep JSON nesting, odd values, a float64
forecast beyond the float32 range the forecasts are kept in) is a PredictorError.
"""

from __future__ import annotations

import itertools
import json
import os
import select
import struct
import time
from dataclasses import dataclass, field

import numpy as np

MAX_HEADER = 16 * 1024**2
FORECAST_DTYPES = ("<f4", "<f8")
CHUNK = 1 << 20


class PredictorError(Exception):
    """The predictor failed, timed out or broke the protocol: the submission is invalid."""


@dataclass
class Window:
    """One forecast window, as the predictor receives it (plus where its forecast goes).

    `t0` is the index, in its series, of the first forecast step; the context `target` covers
    the steps t0 - len(target) .. t0 - 1. `slot` is the window's row in its dataset's forecast
    array (the upstream answer order). `timeline` and `clock` (the forecast start, comparable
    within the timeline, e.g. in ns) order windows across series; by default each series is its
    own timeline, ordered by t0."""

    dataset: str
    item_id: str
    series: int
    t0: int
    start: str
    freq: str
    prediction_length: int
    target: np.ndarray
    out_shape: tuple
    slot: int
    meta: dict = field(default_factory=dict)
    timeline: str | None = None
    clock: int | None = None

    def header(self) -> dict:
        return {
            "dataset": self.dataset,
            "item_id": self.item_id,
            "start": self.start,
            "freq": self.freq,
            "prediction_length": int(self.prediction_length),
            **self.meta,
        }


def _clock(w: Window) -> int:
    return w.t0 if w.clock is None else w.clock


def schedule(windows: list[Window]) -> list[list[Window]]:
    """Rounds in time order: round r holds the r-th tick (windows with the same forecast start)
    of every timeline (see the module docstring). Raises ValueError if a series' windows do not
    move forward in time, or if one tick holds two windows of a series."""
    by_series: dict[tuple[str, int], list[Window]] = {}
    for w in windows:
        by_series.setdefault((w.dataset, w.series), []).append(w)
    for key, ws in by_series.items():
        ws.sort(key=lambda w: w.t0)
        for a, b in itertools.pairwise(ws):
            if not (b.t0 > a.t0 and _clock(b) > _clock(a)):
                raise ValueError(f"{key}: two windows start at step {a.t0}")
    timelines: dict = {}
    for w in windows:
        key = w.timeline if w.timeline is not None else (w.dataset, w.series)
        timelines.setdefault(key, []).append(w)
    rounds: list[list[Window]] = []
    for ws in timelines.values():
        ws.sort(key=lambda w: (_clock(w), w.dataset, w.series))
        ticks = [list(g) for _, g in itertools.groupby(ws, key=_clock)]
        for r, tick in enumerate(ticks):
            if len(rounds) <= r:
                rounds.append([])
            rounds[r].extend(tick)
    return rounds


def batches(windows: list[Window], max_windows: int, max_values: int) -> list[list[Window]]:
    """The calls of a run: each round split into batches of at most `max_windows` windows and
    `max_values` context values (a single window may exceed the latter)."""
    out = []
    for rnd in schedule(windows):
        cur: list[Window] = []
        size = 0
        for w in rnd:
            n = int(w.target.size)
            if cur and (len(cur) >= max_windows or size + n > max_values):
                out.append(cur)
                cur, size = [], 0
            cur.append(w)
            size += n
        if cur:
            out.append(cur)
    return out


# --------------------------------------------------------------------------- framing


def encode(header: dict, arrays: list[np.ndarray] = ()) -> list:
    """The buffers of one frame (written in order)."""
    head = json.dumps(header, separators=(",", ":")).encode()
    bufs = [np.ascontiguousarray(a) for a in arrays]
    size = 4 + len(head) + sum(b.nbytes for b in bufs)
    return [struct.pack("<QI", size, len(head)), head, *(memoryview(b).cast("B") for b in bufs)]


def decode(payload: bytearray | bytes, specs: list[dict]) -> list[np.ndarray]:
    """Writable arrays for `specs` ({"dtype", "shape"}), taken in order after the header."""
    (hlen,) = struct.unpack_from("<I", payload, 0)
    offset, arrays = 4 + hlen, []
    for s in specs:
        dtype = np.dtype(s["dtype"])
        count = int(np.prod(s["shape"], dtype=np.int64))
        if offset + count * dtype.itemsize > len(payload):
            raise ValueError("the payload is shorter than its header says")
        a = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
        arrays.append(a.reshape([int(n) for n in s["shape"]]))
        offset += count * dtype.itemsize
    if offset != len(payload):
        raise ValueError("the payload is longer than its header says")
    return arrays


def header_of(payload: bytearray | bytes) -> dict:
    (hlen,) = struct.unpack_from("<I", payload, 0)
    if hlen > min(MAX_HEADER, len(payload) - 4):
        raise ValueError("bad header length")
    header = json.loads(bytes(payload[4 : 4 + hlen]))
    if not isinstance(header, dict):
        raise ValueError("the header is not a JSON object")
    return header


class Channel:
    """The grader's end of the pipes to one predictor process, with a wall-clock deadline."""

    def __init__(self, wfile, rfile, deadline: float):
        self.wfd, self.rfd, self.deadline = wfile.fileno(), rfile.fileno(), deadline
        os.set_blocking(self.wfd, False)
        os.set_blocking(self.rfd, False)

    def exchange(self, bufs: list | None, max_reply: int) -> bytearray:
        """Write one frame (if any) while reading one reply frame of at most `max_reply` bytes.
        Writing and reading interleave, so a predictor that writes early cannot deadlock us."""
        out = [memoryview(b).cast("B") for b in bufs or []]
        pos = 0
        head, body, need = bytearray(), None, 8
        poller = select.poll()
        poller.register(self.rfd, select.POLLIN)
        writing = bool(out)
        if writing:
            poller.register(self.wfd, select.POLLOUT)
        while True:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise PredictorError("timed out")
            for fd, ev in poller.poll(min(left, 5.0) * 1000):
                if fd == self.wfd:
                    if ev & (select.POLLERR | select.POLLHUP):
                        raise PredictorError("the predictor closed its input")
                    while out:
                        try:
                            n = os.write(self.wfd, out[0][pos : pos + CHUNK])
                        except BlockingIOError:
                            break
                        except BrokenPipeError:
                            raise PredictorError("the predictor exited") from None
                        pos += n
                        if pos == len(out[0]):
                            out.pop(0)
                            pos = 0
                    if not out and writing:
                        poller.unregister(self.wfd)
                        writing = False
                elif fd == self.rfd:
                    try:
                        data = os.read(self.rfd, min(need, 16 * CHUNK))
                    except BlockingIOError:
                        continue
                    if not data:
                        raise PredictorError("the predictor exited")
                    if body is None:
                        head += data
                        need = 8 - len(head)
                        if need == 0:
                            (size,) = struct.unpack("<Q", head)
                            if size > max_reply or size < 4:
                                raise PredictorError(
                                    f"reply of {size} bytes (at most {max_reply} expected)"
                                )
                            body, need = bytearray(), size
                    else:
                        body += data
                        need -= len(data)
                    if body is not None and need == 0:
                        if writing:  # a reply before the request was read: protocol error
                            raise PredictorError("the predictor replied before reading the batch")
                        return body


def _check_reply(payload: bytearray, batch: list[Window]) -> list[np.ndarray]:
    """The forecasts of a reply to `batch`. The reply comes from the predictor, so any failure
    to read it is a PredictorError (json.loads raises RecursionError on deep nesting; odd
    values raise TypeError), never an error of the grader."""
    try:
        return _forecasts(payload, batch)
    except PredictorError:
        raise
    except Exception as e:
        raise PredictorError(f"unreadable reply: {type(e).__name__}: {str(e)[:500]}") from None


def _forecasts(payload: bytearray, batch: list[Window]) -> list[np.ndarray]:
    header = header_of(payload)
    if header.get("type") == "error":
        raise PredictorError(f"predict() failed: {str(header.get('error'))[-3000:]}")
    specs = header.get("arrays")
    if header.get("type") != "forecasts" or not isinstance(specs, list):
        raise PredictorError("unexpected reply")
    if len(specs) != len(batch):
        raise PredictorError(f"{len(specs)} forecasts for {len(batch)} windows")
    for s, w in zip(specs, batch, strict=True):
        if not isinstance(s, dict) or s.get("dtype") not in FORECAST_DTYPES:
            raise PredictorError("forecasts must be float32 or float64 arrays")
        if not isinstance(s.get("shape"), list) or tuple(s["shape"]) != tuple(w.out_shape):
            raise PredictorError(
                f"{w.dataset} {w.item_id}: forecast of shape {s.get('shape')!r:.100}, "
                f"expected {tuple(w.out_shape)}"
            )
    arrays = decode(payload, specs)
    for a, w in zip(arrays, batch, strict=True):
        if not np.isfinite(a).all():
            raise PredictorError(f"{w.dataset} {w.item_id}: forecast with NaN or infinity")
        # The forecasts are kept as float32: a float64 beyond its range would become inf.
        with np.errstate(over="ignore"):
            if a.dtype != np.float32 and not np.isfinite(a.astype(np.float32)).all():
                raise PredictorError(
                    f"{w.dataset} {w.item_id}: forecast outside the float32 range "
                    f"(|value| > {float(np.finfo(np.float32).max):.4g})"
                )
    return arrays


def run(
    channel: Channel,
    windows: list[Window],
    sink,
    max_windows: int = 1024,
    max_values: int = 32 * 1024**2,
    progress=None,
) -> dict:
    """Send every window, in rounds, and pass each forecast to `sink(window, array)`.
    Raises PredictorError if the predictor fails, breaks the protocol or runs out of time."""
    started = time.monotonic()
    ready = _ready(channel)
    calls = batches(windows, max_windows, max_values)
    done = 0
    for k, batch in enumerate(calls):
        arrays = [w.target.astype(np.float32, copy=False) for w in batch]
        header = {
            "type": "batch",
            "call": k,
            "windows": [
                {**w.header(), "target": {"dtype": "<f4", "shape": list(a.shape)}}
                for w, a in zip(batch, arrays, strict=True)
            ],
        }
        max_reply = 4 + MAX_HEADER + sum(8 * int(np.prod(w.out_shape)) for w in batch)
        try:
            reply = channel.exchange(encode(header, arrays), max_reply)
        except PredictorError as e:
            where = f"call {k + 1} of {len(calls)}, {done} windows done"
            raise PredictorError(f"{e} ({where})") from None
        for w, a in zip(batch, _check_reply(reply, batch), strict=True):
            sink(w, a)
        done += len(batch)
        if progress:
            progress(k + 1, len(calls), done, len(windows))
    _send_end(channel)  # best effort: the process is killed afterwards anyway
    return {
        "calls": len(calls),
        "windows": done,
        "startup_s": round(ready, 3),
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def _ready(channel: Channel) -> float:
    t = time.monotonic()
    try:
        payload = channel.exchange(None, 4 + MAX_HEADER)
    except PredictorError as e:
        raise PredictorError(f"{e} while loading predict.py") from None
    try:  # the predictor's message: any failure to read it is the predictor's
        header = header_of(payload)
        kind = header.get("type")
        error = str(header.get("error"))[-3000:] if kind == "error" else ""
    except Exception as e:
        raise PredictorError(
            f"unreadable message while loading predict.py: {type(e).__name__}: {str(e)[:500]}"
        ) from None
    if kind == "error":
        raise PredictorError(f"loading predict.py failed: {error}")
    if kind != "ready":
        raise PredictorError("unexpected message while loading predict.py")
    return time.monotonic() - t


def _send_end(channel: Channel) -> None:
    """Write the end message without waiting for a reply."""
    bufs = [memoryview(b).cast("B") for b in encode({"type": "end"})]
    for b in bufs:
        pos = 0
        while pos < len(b):
            if time.monotonic() > channel.deadline:
                return
            try:
                pos += os.write(channel.wfd, b[pos:])
            except BlockingIOError:
                select.select([], [channel.wfd], [], 1.0)
            except (BrokenPipeError, OSError):
                return
