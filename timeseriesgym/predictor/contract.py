"""The format of a predictor submission: a directory with predict.py, checked without running it.

Used by the grader before it runs a predictor (on the copy it runs), by validate_submission
(which must never import agent code: it runs as root in the agent's container), and by the
local check. Standalone: standard library only.
"""

from __future__ import annotations

import ast
import os
import stat
from pathlib import Path

ENTRY = "predict.py"
FUNCTION = "predict"
MAX_BYTES = 2 * 1024**3  # the whole directory
MAX_ENTRIES = 10_000  # files and directories
MAX_DEPTH = 32
MAX_PYTHON_BYTES = 16 * 1024**2  # predict.py; model weights go into separate files


class ContractError(Exception):
    """The predictor directory does not have the required format."""


def parse_python(path: Path) -> ast.Module:
    """The AST of predict.py, at most MAX_PYTHON_BYTES. Whatever the source makes the parser
    raise is a format error: CPython raises MemoryError ("Parser stack overflowed") and
    RecursionError on a few hundred KB of nesting, so here neither means memory pressure."""
    with open(path, "rb") as f:
        source = f.read(MAX_PYTHON_BYTES + 1)
    if len(source) > MAX_PYTHON_BYTES:
        raise ContractError(
            f"{path.name} is larger than {MAX_PYTHON_BYTES} bytes; keep data such as model "
            "weights in separate files"
        )
    try:
        return ast.parse(source, filename=path.name)
    except SyntaxError as e:
        raise ContractError(f"{path.name} is not valid Python: {e}") from None
    except (ValueError, MemoryError, RecursionError) as e:
        raise ContractError(
            f"{path.name} is not valid Python: the parser gave up ({type(e).__name__})"
        ) from None


def _target_binds(target: ast.expr, name: str) -> bool:
    """An assignment target that binds `name`: the name itself, or inside a tuple, list or
    starred target (`predict, _ = build()`)."""
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, ast.Starred):
        return _target_binds(target.value, name)
    if isinstance(target, ast.Tuple | ast.List):
        return any(_target_binds(t, name) for t in target.elts)
    return False


_BLOCKS = (
    ast.If, ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While,
    ast.Match,
)  # fmt: skip


def binds(body: list, name: str) -> bool:
    """A module-level statement (also inside if/try/with/for/while/match) that binds `name`:
    a def or class, an assignment, or an import (`from model import predict`)."""
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                return True
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_target_binds(t, name) for t in targets):
                return True
        elif isinstance(node, ast.Import | ast.ImportFrom):
            if any((a.asname or a.name.split(".")[0]) == name for a in node.names):
                return True
        elif isinstance(node, _BLOCKS):
            if isinstance(node, ast.For | ast.AsyncFor) and _target_binds(node.target, name):
                return True
            blocks = [getattr(node, "body", []), getattr(node, "orelse", [])]
            blocks += [getattr(node, "finalbody", [])]
            blocks += [h.body for h in getattr(node, "handlers", [])]
            blocks += [case.body for case in getattr(node, "cases", [])]
            if any(binds(b, name) for b in blocks):
                return True
    return False


def check_entry(path: Path) -> None:
    """predict.py: a regular file that parses and binds `predict` at module level."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise ContractError(f"missing file: {path.name}") from None
    if not stat.S_ISREG(st.st_mode):
        raise ContractError(f"{path.name} must be a regular file (not a symlink)")
    if not binds(parse_python(path).body, FUNCTION):
        raise ContractError(f"{path.name} must define a module-level function `{FUNCTION}`")


def entries(path: Path, max_bytes: int = MAX_BYTES, max_entries: int = MAX_ENTRIES) -> list:
    """(relative path, is_dir, size) of every entry under a predictor directory, never
    following a symlink. Only directories and regular files are allowed, within the caps."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise ContractError(f"missing directory: {path}") from None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ContractError(f"{path.name} must be a directory (not a symlink or a file)")
    out, total, stack = [], 0, [(path, 0)]
    while stack:
        d, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise ContractError(f"{path.name}: directories nested deeper than {MAX_DEPTH}")
        with os.scandir(d) as it:
            found = sorted(it, key=lambda e: e.name)
        for e in found:
            rel = os.path.relpath(e.path, path)
            st = e.stat(follow_symlinks=False)
            if len(out) >= max_entries:
                raise ContractError(f"{path.name}: more than {max_entries} files and directories")
            if stat.S_ISDIR(st.st_mode):
                out.append((rel, True, 0))
                stack.append((Path(e.path), depth + 1))
            elif stat.S_ISREG(st.st_mode):
                total += st.st_size
                if total > max_bytes:
                    raise ContractError(f"{path.name}: more than {max_bytes} bytes in total")
                out.append((rel, False, st.st_size))
            else:
                raise ContractError(
                    f"{path.name}/{rel}: only regular files and directories are allowed "
                    "(no symlinks, FIFOs or devices)"
                )
    return out


def check_predictor(path: Path, max_bytes: int = MAX_BYTES, max_entries: int = MAX_ENTRIES):
    """A predictor directory: regular files and directories only, within the caps, with a
    predict.py that parses and binds `predict` at module level. Raises ContractError."""
    entries(path, max_bytes, max_entries)
    check_entry(path / ENTRY)


def copy_predictor(
    src: Path, dst: Path, max_bytes: int = MAX_BYTES, max_entries: int = MAX_ENTRIES
) -> None:
    """Copy a predictor directory to `dst` (which must not exist): directories and regular
    files only, opened without following symlinks, within the caps (checked again on what is
    actually read, as the directory may change while it is copied). Check the copy, not the
    original: only the copy is run."""
    found = entries(src, max_bytes, max_entries)
    dst.mkdir()
    budget = max_bytes
    for rel, is_dir, _size in sorted(found):
        if is_dir:
            (dst / rel).mkdir()
            continue
        fd = os.open(src / rel, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as f, open(dst / rel, "xb") as out:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise ContractError(f"{rel} is not a regular file")
            while chunk := f.read(1 << 20):
                budget -= len(chunk)
                if budget < 0:
                    raise ContractError(f"{src.name}: more than {max_bytes} bytes")
                out.write(chunk)
