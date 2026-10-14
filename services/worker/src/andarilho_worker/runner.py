"""Supervised subprocess execution (README section 8, runner requirements).

* argument arrays only, ``shell=False``;
* each child runs in its own process group/session so the whole tree can be
  terminated (POSIX: ``start_new_session`` + ``killpg``; Windows:
  ``CREATE_NEW_PROCESS_GROUP`` + ``taskkill /T``);
* wall-clock timeout, cancellation callback polled while the child runs (the
  heartbeat thread is independent of log output and sets the cancellation flag);
* bounded captured output with signed URLs and credentials redacted;
* a scratch-disk budget polled during execution;
* a minimal allowlisted environment (no database or storage credentials).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ENV_ALLOWLIST = (
    "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "SystemRoot", "COMSPEC", "PATHEXT", "WINDIR",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "LD_LIBRARY_PATH", "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "QT_QPA_PLATFORM",
)  # fmt: skip

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"https?://[^\s\"'<>]*[?&]X-Amz-[^\s\"'<>]*", re.I), "[REDACTED_SIGNED_URL]"),
    (
        re.compile(r"(X-Amz-(?:Signature|Credential|Security-Token)=)[^&\s\"']+", re.I),
        r"\1[REDACTED]",
    ),
    (re.compile(r"((?:Signature|AWSAccessKeyId|Expires)=)[^&\s\"']+"), r"\1[REDACTED]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_ACCESS_KEY]"),
    (
        re.compile(r"((?:aws_)?secret(?:_access)?_key|password|token)(\s*[=:]\s*)\S+", re.I),
        r"\1\2[REDACTED]",
    ),
    (re.compile(r"(postgres(?:ql)?(?:\+\w+)?://[^:/\s]+:)[^@\s]+@"), r"\1[REDACTED]@"),
)


class Redactor:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        for pattern, replacement in _REDACTIONS:
            text = pattern.sub(replacement, text)
        return text


class RunnerError(Exception):
    def __init__(self, message: str, result: RunResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class ProcessTimeout(RunnerError):
    pass


class ScratchBudgetExceeded(RunnerError):
    pass


class ProcessFailed(RunnerError):
    pass


@dataclass(frozen=True)
class RunResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr_tail: str
    duration_seconds: float


class _Tail:
    """Thread that drains a pipe, keeping a bounded head (stdout) or tail (logs)."""

    def __init__(self, stream, limit: int, keep_head: bool) -> None:
        self._stream = stream
        self._limit = limit
        self._keep_head = keep_head
        self._buf = bytearray()
        self.truncated = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(65536)
                if not chunk:
                    break
                self._buf += chunk
                if len(self._buf) > self._limit:
                    self.truncated = True
                    if self._keep_head:
                        del self._buf[self._limit :]
                    else:
                        del self._buf[: len(self._buf) - self._limit]
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        self.thread.join(timeout=5)
        return self._buf.decode("utf-8", errors="replace")


def directory_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def minimal_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
    env.setdefault("LANG", "C.UTF-8")
    if extra:
        env.update(extra)
    return env


def _kill_tree(proc: subprocess.Popen, grace: float) -> None:
    if proc.poll() is not None and sys.platform != "win32":
        # The leader exited; descendants may still be alive in its group.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        return
    if sys.platform == "win32":
        taskkill = shutil.which("taskkill") or "taskkill"
        subprocess.run(  # noqa: S603 - fixed argv, shell=False
            [taskkill, "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)


class ProcessRunner:
    def __init__(
        self,
        *,
        redactor: Redactor | None = None,
        max_stdout_bytes: int = 4 * 1024 * 1024,
        max_log_bytes: int = 64 * 1024,
        poll_interval: float = 0.1,
        disk_poll_interval: float = 2.0,
        kill_grace_seconds: float = 5.0,
    ) -> None:
        self.redactor = redactor or Redactor()
        self.max_stdout_bytes = max_stdout_bytes
        self.max_log_bytes = max_log_bytes
        self.poll_interval = poll_interval
        self.disk_poll_interval = disk_poll_interval
        self.kill_grace_seconds = kill_grace_seconds

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        check_cancelled: Callable[[], None] | None = None,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        scratch: Path | None = None,
        max_scratch_bytes: int | None = None,
        check: bool = True,
    ) -> RunResult:
        """Run ``argv`` to completion under supervision.

        ``check_cancelled`` raises (e.g. :class:`~andarilho_worker.errors.Cancelled`)
        to request termination; the process group is killed and the exception
        re-raised.
        """
        if isinstance(argv, str | bytes):
            raise TypeError("argv must be a sequence of arguments, never a shell string")
        argv = [str(a) for a in argv]
        if check_cancelled:
            check_cancelled()
        if (
            scratch is not None
            and max_scratch_bytes is not None
            and directory_bytes(scratch) > max_scratch_bytes
        ):
            raise ScratchBudgetExceeded("scratch budget exhausted before start")
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        started = time.monotonic()
        log.info("running %s", self.redactor(" ".join(argv[:2])))
        proc = subprocess.Popen(  # noqa: S603 - argv array, shell=False
            argv,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else minimal_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **kwargs,
        )
        out = _Tail(proc.stdout, self.max_stdout_bytes, keep_head=True)
        err = _Tail(proc.stderr, self.max_log_bytes, keep_head=False)
        last_disk = started
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if now - started > timeout:
                    raise ProcessTimeout(f"{argv[0]} exceeded {timeout:.0f}s")
                if check_cancelled:
                    check_cancelled()
                if (
                    scratch is not None
                    and max_scratch_bytes is not None
                    and now - last_disk >= self.disk_poll_interval
                ):
                    last_disk = now
                    if directory_bytes(scratch) > max_scratch_bytes:
                        raise ScratchBudgetExceeded(f"{argv[0]} exceeded the scratch budget")
                time.sleep(self.poll_interval)
        except BaseException:
            _kill_tree(proc, self.kill_grace_seconds)
            out.text()
            err.text()
            raise
        # The leader exited; make sure no descendant keeps running in its group.
        if sys.platform != "win32":
            _kill_tree(proc, 0)
        result = RunResult(
            argv=tuple(argv),
            returncode=proc.returncode,
            stdout=self.redactor(out.text()),
            stderr_tail=self.redactor(err.text()),
            duration_seconds=time.monotonic() - started,
        )
        if result.stderr_tail:
            log.debug("%s stderr tail: %s", argv[0], result.stderr_tail[-2000:])
        if check and result.returncode != 0:
            raise ProcessFailed(f"{argv[0]} exited with status {result.returncode}", result)
        return result


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None
