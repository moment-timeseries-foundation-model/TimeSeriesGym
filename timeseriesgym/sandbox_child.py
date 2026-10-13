"""The first code a grading sandbox process runs (sandbox.py starts it; see there).

Usage: python -I -B sandbox_child.py <job.json> <status fd>

It imports only the standard library until the process is confined: resource limits, no new
privileges, and, where the kernel has Landlock, file-system access limited to the paths the job
lists, no TCP (Landlock ABI 4), and no signals or abstract UNIX sockets out of the sandbox (ABI
6). It then checks that none of the job's protected paths (the answers) can be opened, imports
the job's entry point, and reports on the status pipe, before any agent code is loaded:

    READY {json}     confined and ready: whatever happens next is the submission's doing
    PROBE {json}     a protected path is readable from here, so agent code must not run
    SETUP <text>     the entry point could not be imported (a problem of the grading setup)

The pipe is closed before the entry point runs, so agent code cannot write to it. The entry point
is called as fn(work_dir, job["job"]) and writes its results under work_dir/out. The process ends
with os._exit as soon as it returns, so that threads or atexit handlers left by agent code can
neither keep it alive nor run after the results are written.
"""

import ctypes
import importlib
import json
import os
import resource
import stat
import sys
import traceback
from pathlib import Path

_PR_SET_NO_NEW_PRIVS = 38
_SYS_LANDLOCK_CREATE_RULESET = 444  # the same numbers on every architecture
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1

# File-system rights (include/uapi/linux/landlock.h) and the ABI versions that added them.
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REMOVE_DIR, REMOVE_FILE, MAKE_DIR, MAKE_REG = 1 << 4, 1 << 5, 1 << 7, 1 << 8
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_FS_RIGHTS = {1: (1 << 13) - 1, 2: (1 << 14) - 1, 3: (1 << 15) - 1, 5: (1 << 16) - 1}
_FILE_RIGHTS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV  # all a file rule may hold
_NET_TCP = 0b11  # bind and connect (ABI 4)
_SCOPES = 0b11  # abstract UNIX sockets and signals (ABI 6)

READ = EXECUTE | READ_FILE | READ_DIR
ACCESS = {
    "read": READ,
    "proc": READ_FILE | READ_DIR,
    "dev": READ | WRITE_FILE | TRUNCATE | IOCTL_DEV,
    "shm": READ | WRITE_FILE | TRUNCATE | MAKE_REG | REMOVE_FILE | MAKE_DIR | REMOVE_DIR | REFER,
    "write": (1 << 16) - 1,
}


class _RulesetAttr(ctypes.Structure):
    _fields_ = (
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    )


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1  # __attribute__((packed)) in the kernel's header
    _fields_ = (("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32))


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _syscall(libc, number: int, *args) -> int:
    """syscall(2), with integer arguments passed as C longs: it reads every argument as one."""
    args = [ctypes.c_long(a) if isinstance(a, int) else a for a in args]
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return result


def landlock_abi() -> int:
    """The kernel's Landlock ABI version; 0 where there is none (or seccomp blocks it)."""
    try:
        flags = _LANDLOCK_CREATE_RULESET_VERSION
        return _syscall(_libc(), _SYS_LANDLOCK_CREATE_RULESET, None, 0, flags)
    except (AttributeError, OSError):  # not Linux, or no Landlock
        return 0


def confine(rules: list[list[str]]) -> int:
    """Allow file-system access only as `rules` ([path, kind of ACCESS] pairs) say and deny TCP
    and signals out of the sandbox, as far as the kernel's Landlock ABI goes. Returns the ABI
    version, 0 if there is no Landlock. Paths that do not exist are left out."""
    abi = landlock_abi()
    if abi < 1:
        return 0
    libc = _libc()
    fs = _FS_RIGHTS[max(v for v in _FS_RIGHTS if v <= abi)]
    attr = _RulesetAttr(fs, _NET_TCP if abi >= 4 else 0, _SCOPES if abi >= 6 else 0)
    ruleset = _syscall(
        libc, _SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0
    )
    try:
        for path, kind in rules:
            try:
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                allowed = ACCESS[kind] & fs
                if not stat.S_ISDIR(os.fstat(fd).st_mode):
                    allowed &= _FILE_RIGHTS
                rule = _PathBeneathAttr(allowed, fd)
                rule_type = _LANDLOCK_RULE_PATH_BENEATH
                _syscall(libc, _SYS_LANDLOCK_ADD_RULE, ruleset, rule_type, ctypes.byref(rule), 0)
            finally:
                os.close(fd)
        _syscall(libc, _SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0)
    finally:
        os.close(ruleset)
    return abi


def limit(limits: dict[str, int]) -> None:
    """Resource limits (RLIMIT_* name: bytes or seconds; never above the current hard limit), no
    new privileges, and first place in line for the kernel's OOM killer."""
    for name, value in limits.items():
        kind = getattr(resource, name)
        hard = resource.getrlimit(kind)[1]
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(kind, (value, value))
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    try:
        prctl = _libc().prctl
    except AttributeError:  # not Linux
        return
    if prctl(_PR_SET_NO_NEW_PRIVS, *map(ctypes.c_ulong, (1, 0, 0, 0))) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")


def readable(path: str) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC)
    except OSError:
        return False
    os.close(fd)
    return True


def main(argv: list[str]) -> int:
    with open(argv[1]) as f:
        job = json.load(f)
    status = os.fdopen(int(argv[2]), "wb", buffering=0)

    def report(kind: str, message: str) -> None:
        status.write(f"{kind} {message}".encode(errors="replace"))
        status.close()

    limit(job["limits"])
    abi = confine(job["rules"]) if job["landlock"] else 0
    exposed = [path for path in job["protected"] if readable(path)]
    if exposed and not job["unconfined_ok"]:
        report("PROBE", json.dumps({"landlock": abi, "readable": exposed[:5]}))
        return 3
    sys.path[:0] = [job["package_root"]]
    sys.path += [path for path in job["sys_path"] if path not in sys.path]
    try:
        import timeseriesgym

        module, name = job["entrypoint"].split(":")
        entrypoint = getattr(importlib.import_module(module), name)
        package = os.path.dirname(os.path.realpath(timeseriesgym.__file__))
    except BaseException:
        report("SETUP", traceback.format_exc()[-4000:])
        return 4
    report("READY", json.dumps({"landlock": abi, "package": package, "readable": exposed[:5]}))
    # Agent code runs from here on.
    try:
        entrypoint(Path.cwd(), job["job"])
    except BaseException:
        traceback.print_exc()
        code = 1
    else:
        code = 0
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
