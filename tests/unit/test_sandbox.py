"""The grading sandbox (timeseriesgym/sandbox.py): what it reads back from agent code."""

import io
import os
import shutil

import numpy as np
import pytest

from timeseriesgym import sandbox

# ---------------------------------------------------------------- reading sandbox outputs


def test_sandbox_outputs_are_read_without_following_links(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.json").write_text('{"a": 1}')
    assert sandbox.read_json(tmp_path, "a.json") == {"a": 1}
    assert sandbox.read_json(tmp_path, "absent.json") is None
    (tmp_path / "secret").write_text('{"answers": 1}')
    (out / "link.json").symlink_to(tmp_path / "secret")
    os.mkfifo(out / "fifo.json")
    (out / "big.json").write_text("[" + "0," * sandbox.MAX_JSON_BYTES + "0]")
    (out / "deep.json").write_text("[" * 100 + "]" * 100)
    (out / "bad.json").write_bytes(b"\xff")
    for name in ("link.json", "fifo.json", "big.json", "deep.json", "bad.json"):
        with pytest.raises(ValueError):
            sandbox.read_json(tmp_path, name)
    shutil.move(out, tmp_path / "real-out")
    (tmp_path / "out").symlink_to(tmp_path / "real-out")
    with pytest.raises(ValueError, match="output directory"):
        sandbox.read_json(tmp_path, "a.json")


def test_sandbox_arrays_must_be_plain_numbers(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "a.npy", np.arange(3.0))
    np.save(out / "c.npy", np.arange(3) + 1j)
    np.save(out / "o.npy", np.array([None, 1], dtype=object), allow_pickle=True)
    np.save(out / "s.npy", np.array(["a"]))
    header = io.BytesIO()
    shape = {"descr": "<f8", "fortran_order": False, "shape": (2**60,)}
    np.lib.format.write_array_header_1_0(header, shape)
    (out / "huge.npy").write_bytes(header.getvalue())  # declares 8 EiB and holds none
    assert sandbox.read_npy(tmp_path, "a.npy").tolist() == [0.0, 1.0, 2.0]
    assert sandbox.read_npy(tmp_path, "c.npy").dtype == np.complex128
    for name in ("o.npy", "s.npy", "huge.npy"):
        with pytest.raises(ValueError):
            sandbox.read_npy(tmp_path, name)
