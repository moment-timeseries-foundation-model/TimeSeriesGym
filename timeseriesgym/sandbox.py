"""Run agent-written code and models in a sandbox while grading.

Code challenges (resnet-tensorflow-to-pytorch, stomp-R-to-python) import and run the submitted
`.py` file, and hyperparameter-search challenges unpickle or `torch.jit.load` the submitted
models. All of this runs in a separate process, the sandbox, and never in the grading process,
which holds the answers. Each grader is split in two halves. The sandbox half runs the
submission as the grader always did, up to the point where it reads the answers, and saves what
it computed under `out/` in its work directory: JSON and numeric `.npy` arrays. The trusted half,
in the grading process, reads those files back, size-capped and without following links, and
compares them with the answers. Everything it reads back is untrusted: agent code runs in the
same process as the sandbox half and can rewrite any of it. A sandbox run that fails, times out,
runs out of memory or leaves unusable output makes the submission invalid.

The sandbox process
- runs as the user `sandbox` (TIMESERIESGYM_SANDBOX_USER) when grading runs as root and that
  user exists, as in the grading container; otherwise as the grading user (see below);
- gets a scrubbed environment (no credentials; HOME and TMPDIR are its work directory), a copy
  of the submission and of the public input data in a fresh work directory, and stdin from
  /dev/null;
- runs with limits: wall clock (TIMESERIESGYM_SANDBOX_TIMEOUT_SEC, default per challenge), the
  memory of all its processes together (TIMESERIESGYM_SANDBOX_MEMORY_MB, default 3/4 of the
  machine's or container's memory; measured every 0.1 s), CPU time (never binding before the
  wall clock), 1 GiB per written file, no core dumps, no new privileges (setuid binaries do not
  raise them), and first place for the kernel's OOM killer;
- is confined by Landlock where the kernel has it (Linux 5.13 and later): it can read only the
  system and Python directories, the timeseriesgym package and its work directory, write only
  its work directory and /dev/shm, open no TCP connections (Linux 6.7), and signal no process
  outside the sandbox (Linux 6.12).
Before agent code runs, the sandbox checks that it cannot open the challenge's private files
(the answers and the raw data); if it can, grading stops with a SandboxError. When the run ends,
every process it started is killed, also those that left its session or process group: the
grading process is their subreaper, so none can escape its tree.

What the two modes give. As root with a sandbox user (the grading container): the sandbox can
read only what that user may read and Landlock allows, and can neither signal nor trace the
grading process. As a normal user on a host, there is no other user to switch to, so the sandbox
has the grading user's uid and Landlock alone keeps it from the answers, from the user's home
directory (credentials included) and from other processes. Without Landlock (older kernels,
other operating systems, or a seccomp policy that blocks it), the answers are readable, and code
grading stops with a SandboxError; TIMESERIESGYM_SANDBOX_UNCONFINED=1 runs it anyway, for
submissions that are trusted. Grading as root without a sandbox user is refused: agent code
would run as root. Not covered: memory allocated faster than it is measured (the OOM killer,
which takes sandbox processes first, is the backstop) and files in tmpfs such as /dev/shm,
which count as no process's memory; UDP and UNIX-socket traffic; disk use beyond the per-file
cap (bounded by the time limit); and in host mode, connecting to pathname UNIX sockets that the
user can reach (e.g. Docker's), the user's process quota, and, below Linux 6.12, signals to the
user's other processes. Sandbox runs in one grading process are serialized.
"""

import contextlib
import ctypes
import io
import json
import math
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from timeseriesgym.utils import InvalidSubmissionError, get_logger

logger = get_logger(__name__)

USER_ENV = "TIMESERIESGYM_SANDBOX_USER"
TIMEOUT_ENV = "TIMESERIESGYM_SANDBOX_TIMEOUT_SEC"
MEMORY_ENV = "TIMESERIESGYM_SANDBOX_MEMORY_MB"
UNCONFINED_ENV = "TIMESERIESGYM_SANDBOX_UNCONFINED"
DEFAULT_USER = "sandbox"
FILE_BYTES = 2**30  # RLIMIT_FSIZE: any one file the sandbox writes, its stdout and stderr too
MAX_JSON_BYTES = 2**20
MAX_ARRAY_BYTES = 64 * 2**20
MAX_JSON_DEPTH = 16
MAX_PROBE_FILES = 32  # protected files whose readability is checked, per protected directory
KILL_ROUNDS = 100
POLL_SEC = 0.1  # how often the memory of the sandbox processes is measured
DRAIN_SEC = 30
LOG_CHARS = 4000
LANDLOCK = True  # whether to confine the sandbox with Landlock where the kernel has it

CHILD = Path(__file__).with_name("sandbox_child.py")
PACKAGE_DIR = Path(__file__).resolve().parent
_SYSTEM_DIRS = ("/usr", "/lib", "/lib32", "/lib64", "/libx32", "/bin", "/sbin", "/etc")
_PASSED_ENV = (
    "LANG",
    "LC_ALL",
    "TZ",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_PR_SET_CHILD_SUBREAPER, _PR_GET_CHILD_SUBREAPER = 36, 37
_LOCK = threading.Lock()


class SandboxError(RuntimeError):
    """The sandbox cannot run agent code safely or failed to start: a problem of the grading
    setup, not of the submission. Grading stops instead of scoring the submission invalid."""


# ---------------------------------------------------------------- the trusted half


@dataclass(frozen=True)
class Limits:
    timeout_sec: int
    memory_bytes: int | None

    @classmethod
    def for_challenge(cls, default_timeout_sec: int) -> "Limits":
        """The limits from the environment (MEMORY_ENV 0: no memory limit), else the defaults."""
        try:
            timeout = int(os.environ.get(TIMEOUT_ENV) or default_timeout_sec)
            memory_mb = os.environ.get(MEMORY_ENV)
            memory = int(memory_mb) * 2**20 if memory_mb else _default_memory_bytes()
        except ValueError:
            raise SandboxError(f"{TIMEOUT_ENV} and {MEMORY_ENV} must be integers.") from None
        return cls(timeout_sec=timeout, memory_bytes=memory)


def _default_memory_bytes() -> int | None:
    """3/4 of the memory of the machine, or of the container (cgroup limit) if that is less."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        with contextlib.suppress(OSError, ValueError):
            total = min(total, int(Path(path).read_text().strip()))
    return total * 3 // 4


def sandbox_user() -> tuple[int, int] | None:
    """(uid, gid) the sandbox runs as when grading runs as root; None for the grading user's own.
    Raises SandboxError as root without a sandbox user."""
    if os.geteuid() != 0:
        return None
    name = os.environ.get(USER_ENV) or DEFAULT_USER
    try:
        user = pwd.getpwnam(name)
    except KeyError:
        raise SandboxError(
            f"Grading runs as root, but there is no user {name!r} to run agent code as. Create "
            f"one (e.g. `useradd --system --no-create-home {name}`) or set {USER_ENV}."
        ) from None
    if user.pw_uid == 0:
        raise SandboxError(f"The sandbox user {name!r} is root.")
    return user.pw_uid, user.pw_gid


@dataclass
class Run:
    """A finished sandbox run. Read its outputs inside the `run()` block."""

    work: Path
    returncode: int | None
    exceeded: str | None  # the limit it was stopped at: "time" or "memory"
    limits: Limits
    log: str  # the end of the sandbox's stdout, then of its stderr

    def failure(self) -> str | None:
        if self.exceeded == "time":
            return f"it ran longer than {self.limits.timeout_sec} s"
        if self.exceeded == "memory":
            return f"it used more than {self.limits.memory_bytes // 2**20} MiB of memory"
        if self.returncode != 0:
            how = (
                f"killed by {signal.Signals(-self.returncode).name}"
                if self.returncode is not None and self.returncode < 0
                else f"exit code {self.returncode}"
            )
            return f"it failed ({how})"
        return None

    def result(self, name: str = "result.json") -> Any:
        """The JSON value the sandbox half saved as out/<name>. Raises InvalidSubmissionError if
        the run failed or the file is missing or unusable."""
        failure = self.failure()
        if failure is not None:
            raise InvalidSubmissionError(
                f"Submission invalid! Its code was run in the grading sandbox and {failure}. "
                f"Sandbox log (end):\n{self.log[-1500:]}"
            )
        try:
            value = read_json(self.work, name)
        except ValueError as e:
            raise InvalidSubmissionError(f"Submission invalid! {e}") from None
        if value is None:
            raise InvalidSubmissionError(
                f"Submission invalid! The grading sandbox produced no results. Sandbox log "
                f"(end):\n{self.log[-1500:]}"
            )
        return value

    def array(self, name: str):
        """The numeric array the sandbox half saved as out/<name>, or None if it saved none.
        Raises InvalidSubmissionError if the file is unusable."""
        try:
            return read_npy(self.work, name)
        except ValueError as e:
            raise InvalidSubmissionError(f"Submission invalid! {e}") from None


@contextlib.contextmanager
def run(
    entrypoint: str,
    job: dict,
    limits: Limits,
    agent_files: dict[str, Path] | None = None,
    data: dict[str, Path] | None = None,
    protected: Iterable[Path] = (),
) -> Iterator[Run]:
    """Runs `entrypoint` ("module:function", called as function(work_dir, job)) in a sandbox.

    `agent_files` (name in the work directory: path) are the submission's files: each must be a
    regular file, not a link (else InvalidSubmissionError); missing ones are left out. `data`
    are trusted directories or files copied in as they are, e.g. the public input data.
    `protected` are the challenge's private paths: the sandbox must be unable to read them.
    Yields the finished Run, whose outputs can be read until the block ends.
    """
    ids = sandbox_user()
    protected = list(protected)
    with _LOCK:
        root = Path(tempfile.mkdtemp(prefix="timeseriesgym-sandbox-"))
        try:
            work = _work_dir(root, ids, agent_files or {}, data or {})
            job_file = root / "job.json"
            job_file.write_text(
                json.dumps(
                    {
                        "entrypoint": entrypoint,
                        "job": job,
                        "limits": _rlimits(limits),
                        "landlock": LANDLOCK,
                        "rules": _landlock_rules(root, work, protected),
                        "protected": _probe_paths(protected),
                        "unconfined_ok": os.environ.get(UNCONFINED_ENV) == "1",
                        "package_root": str(root / "package"),
                        "sys_path": _extra_sys_path(protected),
                    }
                )
            )
            os.chmod(job_file, 0o644)
            yield _start(root, work, job_file, ids, limits)
        finally:
            _remove(root)


def _work_dir(
    root: Path, ids: tuple[int, int] | None, agent_files: dict[str, Path], data: dict[str, Path]
) -> Path:
    os.chmod(root, 0o755 if ids else 0o700)
    (root / "package").mkdir()
    # Only the package is importable from here: the directory it is in may hold other things
    # (an editable checkout holds the tests, or the data).
    (root / "package" / "timeseriesgym").symlink_to(PACKAGE_DIR)
    (root / "logs").mkdir(mode=0o700)
    work = root / "work"
    (work / "tmp").mkdir(parents=True)
    for name, src in agent_files.items():
        _copy_agent_file(Path(src), work / name)
    for name, src in data.items():
        if Path(src).is_dir():
            shutil.copytree(src, work / name, symlinks=False)
        else:
            shutil.copy(src, work / name)
    if ids:
        for path, _, files in os.walk(work):
            for name in [".", *files]:
                os.lchown(os.path.join(path, name), *ids)
            os.chmod(path, 0o700)
    return work


def _copy_agent_file(src: Path, dst: Path) -> None:
    try:
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return  # the sandbox half reports a missing file as the native grader did
    except OSError as e:
        raise InvalidSubmissionError(
            f"Submission invalid! {src} cannot be read without following a link: {e.strerror}"
        ) from None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise InvalidSubmissionError(f"Submission invalid! {src} is not a regular file.")
        with open(dst, "xb") as out:
            shutil.copyfileobj(f, out, 2**20)


def _rlimits(limits: Limits) -> dict[str, int]:
    # Memory is limited by _wait, over all sandbox processes together: an address-space or data
    # limit per process would count what thread pools and CUDA only reserve (2.7 GB of data
    # mappings once torch is imported on 64 cores).
    cpu = limits.timeout_sec * max(8, os.cpu_count() or 1)  # never before the wall clock
    return {"RLIMIT_CPU": cpu, "RLIMIT_FSIZE": FILE_BYTES, "RLIMIT_CORE": 0}


def _related(path: Path, protected: list[Path]) -> bool:
    return any(path == p or path in p.parents or p in path.parents for p in protected)


def _resolved(paths: Iterable) -> list[Path]:
    out = []
    for path in paths:
        with contextlib.suppress(OSError, RuntimeError):
            out.append(Path(path).resolve())
    return out


def _python_paths() -> list[Path]:
    """The directories of the Python installation and its packages, as the grading process sees
    them, except its working directory."""
    import site
    import sysconfig

    cwd = Path.cwd().resolve()
    paths = [sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix]
    paths += sysconfig.get_paths().values()
    paths += site.getsitepackages()
    if site.ENABLE_USER_SITE:
        paths.append(site.getusersitepackages())
    paths += [p for p in sys.path if p]
    return [p for p in _resolved(paths) if p != cwd and p.exists()]


def _extra_sys_path(protected: Iterable[Path]) -> list[str]:
    """The grading process's import path, appended to the sandbox's (python -I leaves out the
    user site directory and PYTHONPATH), so that agent code can import what it could before."""
    keep = _resolved(protected)
    cwd = Path.cwd().resolve()
    extra = []
    for path in sys.path:
        resolved = _resolved([path]) if path else []
        if resolved and resolved[0].exists() and resolved[0] != cwd:
            if not _related(resolved[0], keep):
                extra.append(path)
    return extra


def _landlock_rules(root: Path, work: Path, protected: Iterable[Path]) -> list[tuple[str, str]]:
    """What the sandbox may access. A directory that holds a protected path, or lies inside
    one, is left out (the probe then decides whether grading can go on)."""
    keep = _resolved(protected)
    read = [Path(p) for p in _SYSTEM_DIRS] + _python_paths() + [PACKAGE_DIR, root / "package"]
    rules = [(str(p), "read") for p in dict.fromkeys(read) if not _related(p, keep)]
    rules += [("/proc", "proc"), ("/sys", "proc"), ("/dev", "dev"), ("/dev/shm", "shm")]
    rules.append((str(work), "write"))
    return rules


def _probe_paths(protected: Iterable[Path]) -> list[str]:
    """The protected paths, and for each directory up to MAX_PROBE_FILES files inside it: file
    permissions can differ from those of the directory."""
    probe = []
    for path in map(Path, protected):
        if not path.exists():
            continue
        probe.append(str(path))
        if path.is_dir():
            found = []
            for parent, dirs, files in os.walk(path):
                dirs.sort()
                found += [os.path.join(parent, f) for f in sorted(files)]
                if len(found) >= MAX_PROBE_FILES:
                    break
            probe += found[:MAX_PROBE_FILES]
    return list(dict.fromkeys(probe))


def _child_env(work: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASSED_ENV if k in os.environ}
    env.update(
        PATH=os.pathsep.join(
            [str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin"]
        ),
        HOME=str(work),
        TMPDIR=str(work / "tmp"),
        MPLCONFIGDIR=str(work / "tmp"),
        XDG_CACHE_HOME=str(work / "tmp"),
    )
    return env


def _start(root: Path, work: Path, job_file: Path, ids, limits: Limits) -> Run:
    # Output goes to files in a directory the sandbox cannot reach: RLIMIT_FSIZE caps them, only
    # their ends are read, and a process that keeps them open cannot stall the grader.
    logs = root / "logs"
    with open(logs / "stdout", "w+b") as out, open(logs / "stderr", "w+b") as err:
        read_fd, write_fd = os.pipe()
        # Processes orphaned below the grading process become its children instead of init's,
        # so that sandbox processes stay findable whatever session they move to.
        was_subreaper = _set_subreaper(True)
        try:
            proc, before = _popen(work, job_file, ids, out, err, write_fd)
            exceeded = _wait(proc, before, limits)
            status = _read_status(read_fd)
        finally:
            os.close(read_fd)
            if not was_subreaper:
                _set_subreaper(False)
        log = _clean(_tail(out) + "\n" + _tail(err))  # stderr last: tracebacks end the log
    _check_status(status, ids, log)
    if log:
        logger.info(f"Grading sandbox log (end):\n{log[-LOG_CHARS:]}")
    return Run(work, proc.returncode, exceeded, limits, log)


def _popen(work: Path, job_file: Path, ids, out, err, write_fd: int):
    """Starts the sandbox; returns (process, children of this process from before)."""
    user = {"user": ids[0], "group": ids[1], "extra_groups": []} if ids else {}
    before = {pid for pid, parent in _children().items() if parent == os.getpid()}
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", str(CHILD), str(job_file), str(write_fd)],
            cwd=work,
            env=_child_env(work),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            pass_fds=(write_fd,),
            start_new_session=True,
            **user,
        )
    except OSError as e:
        raise SandboxError(f"The grading sandbox could not be started: {e}") from e
    finally:
        os.close(write_fd)
    return proc, before


def _wait(proc: subprocess.Popen, before: set[int], limits: Limits) -> str | None:
    """Waits for the sandbox, within its time and memory limits, then kills every process it
    started. Returns the limit it exceeded ("time" or "memory"), if any."""
    deadline = time.monotonic() + limits.timeout_sec
    exceeded = None
    try:
        while True:
            try:
                proc.wait(timeout=POLL_SEC)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() > deadline:
                exceeded = "time"
                break
            if limits.memory_bytes and _memory(_tree(before)) > limits.memory_bytes:
                exceeded = "memory"
                break
    finally:
        _kill_tree(before, proc.pid)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=DRAIN_SEC)
    return exceeded


def _memory(pids: list[int]) -> int:
    """The memory the processes use together: their proportional set sizes (shared pages
    split between the processes that share them), or the resident set size where the PSS
    cannot be read."""
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/smaps_rollup", "rb") as f:
                for line in f:
                    if line.startswith(b"Pss:"):
                        total += int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError):
            with contextlib.suppress(OSError, ValueError, IndexError):
                with open(f"/proc/{pid}/statm", "rb") as f:
                    total += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return total


def _check_status(status: str, ids, log: str) -> None:
    """Raises SandboxError unless the sandbox was confined and ready before agent code ran."""
    kind, _, message = status.partition(" ")
    if kind == "PROBE":
        info = json.loads(message)
        raise SandboxError(
            f"Agent code would be able to read the challenge's private files from the grading "
            f"sandbox (e.g. {info['readable'][0]}), so it is not run. "
            + (
                "Make them unreadable to the sandbox user, or grade where Landlock is available."
                if ids
                else f"Landlock (Linux 5.13 or later) is {'' if info['landlock'] else 'not '}"
                f"available here. Grade in a container as root with a sandbox user, or set "
                f"{UNCONFINED_ENV}=1 to run trusted submissions without this protection."
            )
        )
    if kind != "READY":
        detail = message if kind == "SETUP" else log[-LOG_CHARS:]
        raise SandboxError(f"The grading sandbox failed to start:\n{detail}")
    info = json.loads(message)
    if Path(info["package"]) != PACKAGE_DIR:
        raise SandboxError(
            f"The grading sandbox imported timeseriesgym from {info['package']}, not from "
            f"{PACKAGE_DIR}."
        )
    if info["readable"]:
        logger.warning(
            f"{UNCONFINED_ENV}=1: agent code ran with read access to private files, e.g. "
            f"{info['readable'][0]}."
        )


def _read_status(fd: int) -> str:
    """What the sandbox reported before agent code ran (it closed its end then)."""
    os.set_blocking(fd, False)
    data = b""
    with contextlib.suppress(OSError):
        while len(data) < 2**16:
            chunk = os.read(fd, 2**16)
            if not chunk:
                break
            data += chunk
    return data.decode(errors="replace")


def _tail(f, n: int = 20000) -> str:
    f.seek(max(0, os.fstat(f.fileno()).st_size - n))
    return f.read(n).decode(errors="replace")


def _clean(text: str) -> str:
    """Agent output for the grading log, without control characters (terminal escapes)."""
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "?", text).strip()


def _set_subreaper(on: bool) -> bool:
    """Sets whether this process is the child subreaper; returns the previous setting."""
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except AttributeError:  # not Linux
        return False
    previous = ctypes.c_int(0)
    prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(previous), 0, 0, 0)
    prctl(_PR_SET_CHILD_SUBREAPER, ctypes.c_ulong(int(on)), 0, 0, 0)
    return bool(previous.value)


def _proc_stat(pid: str) -> tuple[str, int] | None:
    """(state, parent pid) of a process, None if it is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    fields = data[data.rfind(b")") + 2 :].split()  # the name, in parentheses, may hold anything
    return fields[0].decode(), int(fields[1])


def _children() -> dict[int, int]:
    """pid -> parent pid, for every process."""
    parents = {}
    with contextlib.suppress(OSError):
        for entry in os.scandir("/proc"):
            if entry.name.isdigit():
                info = _proc_stat(entry.name)
                if info:
                    parents[int(entry.name)] = info[1]
    return parents


def _running(pid: int) -> bool:
    """Whether any thread of the process can still run: the leader of a thread group whose main
    thread has exited shows state Z while its other threads go on."""
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return False
    return any((_proc_stat(f"{pid}/task/{tid}") or ("X",))[0] not in "ZX" for tid in tids)


def _descendants(exclude: set[int]) -> list[int]:
    """This process's descendants, except the children in `exclude` and theirs."""
    parents = _children()
    tree: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        tree.setdefault(parent, []).append(pid)
    stack = [p for p in tree.get(os.getpid(), []) if p not in exclude]
    found = []
    while stack:
        pid = stack.pop()
        found.append(pid)
        stack += tree.get(pid, [])
    return found


def _tree(exclude: set[int]) -> list[int]:
    """_descendants(exclude), from the children lists of the processes' threads: cheaper than
    reading the status of every process, but it can miss processes created meanwhile."""
    me = os.getpid()
    if not os.path.exists(f"/proc/{me}/task/{me}/children"):  # no CONFIG_PROC_CHILDREN
        return _descendants(exclude)
    found, stack = [], [me]
    while stack:
        pid = stack.pop()
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue  # it exited meanwhile
        children = set()
        for tid in tids:
            with contextlib.suppress(OSError, ValueError):
                children.update(
                    map(int, Path(f"/proc/{pid}/task/{tid}/children").read_text().split())
                )
        if pid == me:
            children -= exclude
        found += children
        stack += children
    return found


def _kill_tree(before: set[int], child: int) -> None:
    """SIGKILL every process the sandbox started until none can run, and reap those that became
    this process's children (except `child`, which Popen reaps). A process with a fatal signal
    pending cannot fork, so each round catches the few forked just before it."""
    if not os.path.isdir("/proc"):  # not Linux: the session is all that can be found
        with contextlib.suppress(OSError):
            os.killpg(child, signal.SIGKILL)
        return
    for round_ in range(KILL_ROUNDS):
        pids = _descendants(before)
        if round_ and not any(_running(p) for p in pids):
            break
        for pid in pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        for pid in pids:
            if pid != child:
                with contextlib.suppress(OSError):
                    os.waitpid(pid, os.WNOHANG)
        time.sleep(0.01)
    else:
        raise SandboxError("Processes started by agent code survive SIGKILL.")
    for pid in _descendants(before):
        if pid != child:
            with contextlib.suppress(OSError):
                os.waitpid(pid, os.WNOHANG)


def _remove(root: Path) -> None:
    """Removes a sandbox's directory, whatever agent code left in it: directories it took the
    permissions of, or nested too deep for shutil.rmtree, which recurses."""
    try:
        for path, dirs, _ in os.walk(root):
            for name in dirs:
                full = os.path.join(path, name)
                with contextlib.suppress(OSError):
                    if not os.path.islink(full):
                        os.chmod(full, 0o700)
        shutil.rmtree(root)
    except (OSError, RecursionError):
        subprocess.run(["rm", "-rf", "--", str(root)], check=False)
    if os.path.lexists(root):
        logger.warning(f"Could not remove the grading sandbox's directory {root}.")


# ---------------------------------------------------------------- reading sandbox outputs


def read_output(work: Path, name: str, max_bytes: int) -> bytes | None:
    """The bytes of work/out/<name>, None if it is absent. Neither out/ nor the file may be a
    link, and FIFOs, devices and files over `max_bytes` raise ValueError. Call this only after
    the run, when no sandbox process is left to swap the files."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        dfd = os.open(work / "out", flags | os.O_DIRECTORY)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ValueError(f"unusable sandbox output directory: {e.strerror}") from None
    try:
        try:
            fd = os.open(name, flags | os.O_NONBLOCK, dir_fd=dfd)
        except FileNotFoundError:
            return None
        except OSError as e:
            raise ValueError(f"unusable sandbox output {name}: {e.strerror}") from None
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
                raise ValueError(f"sandbox output {name} is not a regular file of <= {max_bytes} B")
            data = f.read(max_bytes + 1)
    finally:
        os.close(dfd)
    if len(data) > max_bytes:
        raise ValueError(f"sandbox output {name} is larger than {max_bytes} B")
    return data


def read_json(work: Path, name: str, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """The JSON value in work/out/<name>, None if the file is absent. Raises ValueError if it is
    unusable, does not parse, or nests deeper than MAX_JSON_DEPTH."""
    data = read_output(work, name, max_bytes)
    if data is None:
        return None
    try:
        value = json.loads(data)
    except Exception as e:  # RecursionError on deep nesting, UnicodeDecodeError on bad bytes
        raise ValueError(f"sandbox output {name} is not JSON ({type(e).__name__})") from None
    stack = [(value, 1)]
    while stack:
        v, depth = stack.pop()
        if isinstance(v, dict | list):
            if depth > MAX_JSON_DEPTH:
                raise ValueError(f"sandbox output {name} nests deeper than {MAX_JSON_DEPTH} levels")
            stack += [(x, depth + 1) for x in (v.values() if isinstance(v, dict) else v)]
    return value


def read_npy(work: Path, name: str, max_bytes: int = MAX_ARRAY_BYTES):
    """The numeric array in work/out/<name> (.npy format 1.0 or 2.0), None if it is absent.
    Raises ValueError for anything else. The header is checked against the data before NumPy
    reads it, since np.load first allocates what the header declares."""
    import numpy as np

    data = read_output(work, name, max_bytes)
    if data is None:
        return None
    try:
        f = io.BytesIO(data)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"unsupported .npy format version {version}")
        if dtype.kind not in "biufc" or dtype.names is not None or dtype.subdtype is not None:
            raise ValueError(f"dtype {dtype} is not numeric")
        if f.tell() + math.prod(shape) * dtype.itemsize != len(data):
            raise ValueError(f"the data do not match the header (shape {shape}, {dtype})")
        f.seek(0)
        array = np.load(f, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"sandbox output {name} is not a numeric .npy array ({e})") from None
    if not isinstance(array, np.ndarray):
        raise ValueError(f"sandbox output {name} is not a .npy array")
    return array


def scores(result: Any, keys: Iterable[str]) -> dict[str, str | None]:
    """The structural scores ("k/n" strings or None) a code grader's sandbox half saved under
    "scores". Raises InvalidSubmissionError if they are not that."""
    found = result.get("scores") if isinstance(result, dict) else None
    keys = list(keys)
    if not isinstance(found, dict) or set(found) != set(keys):
        raise InvalidSubmissionError("Submission invalid! Unusable scores from the sandbox.")
    for key in keys:
        if not (found[key] is None or (isinstance(found[key], str) and len(found[key]) <= 32)):
            raise InvalidSubmissionError("Submission invalid! Unusable scores from the sandbox.")
    return {key: found[key] for key in keys}


# ---------------------------------------------------------------- helpers for sandbox halves


def run_code(
    entrypoint: str,
    submission_file_path: Path,
    coding_config: dict,
    private_paths: Iterable[Path] | None,
    default_timeout_sec: int = 1800,
):
    """run() for a code challenge: the submission file, under its own name, and a copy of the
    public input data (`coding_config["input_data_dir"]`, as `data/`). The sandbox half gets the
    config back from job_coding_config()."""
    config = dict(coding_config)
    input_dir = Path(config.pop("input_data_dir"))
    job = {"submission": Path(submission_file_path).name, "coding_config": config}
    return run(
        entrypoint,
        job,
        Limits.for_challenge(default_timeout_sec),
        agent_files={job["submission"]: Path(submission_file_path)},
        data={"data": input_dir},
        protected=private_paths or (),
    )


def job_coding_config(work: Path, job: dict) -> dict:
    """The coding config of a run_code() job, with the input data directory in the sandbox."""
    return {**job["coding_config"], "input_data_dir": work / "data"}


def load_module(path: Path):
    """Imports a submitted file as the code graders always have (in the sandbox only)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(work: Path, name: str, value: Any) -> None:
    (work / "out").mkdir(exist_ok=True)
    (work / "out" / name).write_text(json.dumps(value))


def save_array(work: Path, name: str, value: Any) -> None:
    """Saves an array for the trusted half as work/out/<name>. An object array of real numbers
    is saved as float64, which NumPy computes with as it did with the objects; raises ValueError
    for anything else that is not numeric."""
    import numbers

    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind == "O":
        if not all(isinstance(x, numbers.Real) for x in array.flat):
            raise ValueError("the outputs are not numeric")
        array = array.astype(np.float64)
    if array.dtype.kind not in "biufc" or array.dtype.names is not None:
        raise ValueError(f"the outputs have dtype {array.dtype}, which is not numeric")
    (work / "out").mkdir(exist_ok=True)
    np.save(work / "out" / name, array, allow_pickle=False)
