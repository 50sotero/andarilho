from __future__ import annotations

import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from andarilho_worker.errors import Cancelled
from andarilho_worker.runner import (
    ProcessFailed,
    ProcessRunner,
    ProcessTimeout,
    Redactor,
    ScratchBudgetExceeded,
)

PARENT = textwrap.dedent(
    """
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(sys.argv[1], "w") as fh:
        fh.write(str(child.pid))
    print("spawned", flush=True)
    time.sleep(120)
    """
)


def alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # reaped zombies are gone; unreaped ones show state Z
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except FileNotFoundError:
        return True


def wait_for_file(path: Path, timeout: float = 15) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError("grandchild pid never written")


def wait_dead(pid: int, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def runner() -> ProcessRunner:
    return ProcessRunner(kill_grace_seconds=1.0, poll_interval=0.05)


def test_cancel_terminates_process_group(runner, tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    pid_file = tmp_path / "grandchild.pid"
    flag = threading.Event()

    def check() -> None:
        if flag.is_set():
            raise Cancelled("cancel")

    def cancel_later() -> None:
        wait_for_file(pid_file)
        flag.set()

    threading.Thread(target=cancel_later, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(Cancelled):
        runner.run([sys.executable, str(script), str(pid_file)], timeout=60, check_cancelled=check)
    assert time.monotonic() - started < 30
    assert wait_dead(wait_for_file(pid_file))


def test_timeout_terminates_process_group(runner, tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    pid_file = tmp_path / "grandchild.pid"
    with pytest.raises(ProcessTimeout):
        runner.run([sys.executable, str(script), str(pid_file)], timeout=3)
    assert wait_dead(wait_for_file(pid_file))


def test_rejects_shell_strings(runner):
    with pytest.raises(TypeError):
        runner.run("echo hi", timeout=5)  # type: ignore[arg-type]


def test_output_is_bounded_and_redacted(tmp_path):
    secret_url = (
        "https://bucket.s3.amazonaws.com/tenants/x/a.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=AKIAABCDEFGHIJKLMNOP%2F20261014&X-Amz-Signature=deadbeef"
    )
    code = (
        "import sys\n"
        f"print({secret_url!r})\n"
        "print('key AKIAABCDEFGHIJKLMNOP and secret_access_key=supersecretvalue')\n"
        "print('db postgresql://user:hunter22@host/db')\n"
        "sys.stderr.write('X-Amz-Signature=abc123&other=1\\n' + 'x' * 200000)\n"
        "print('custom-secret-token-value')\n"
    )
    runner = ProcessRunner(redactor=Redactor(["custom-secret-token-value"]), max_log_bytes=1024)
    result = runner.run([sys.executable, "-c", code], timeout=30)
    out = result.stdout
    assert "deadbeef" not in out and "X-Amz-Credential=AKIA" not in out
    assert "[REDACTED_SIGNED_URL]" in out
    assert "AKIAABCDEFGHIJKLMNOP" not in out and "supersecretvalue" not in out
    assert "hunter22" not in out
    assert "custom-secret-token-value" not in out
    assert len(result.stderr_tail) <= 1024


def test_redactor_handles_bare_query_parameters():
    text = Redactor()("GET ?X-Amz-Signature=abcdef&X-Amz-Credential=AKIA/x&Expires=1")
    assert "abcdef" not in text and "AKIA/x" not in text


def test_nonzero_exit_raises(runner):
    with pytest.raises(ProcessFailed) as info:
        runner.run([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=30)
    assert info.value.result.returncode == 3


def test_scratch_budget_enforced(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    code = "import sys, time\nopen(sys.argv[1], 'wb').write(b'0' * 200000)\ntime.sleep(60)\n"
    runner = ProcessRunner(disk_poll_interval=0.1, kill_grace_seconds=1.0)
    with pytest.raises(ScratchBudgetExceeded):
        runner.run(
            [sys.executable, "-c", code, str(scratch / "big.bin")],
            timeout=60,
            scratch=scratch,
            max_scratch_bytes=100000,
        )


def test_child_environment_excludes_credentials(monkeypatch):
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "do-not-leak")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/d")
    code = (
        "import os; print(os.environ.get('S3_SECRET_ACCESS_KEY'), os.environ.get('DATABASE_URL'))"
    )
    result = ProcessRunner().run([sys.executable, "-c", code], timeout=30)
    assert result.stdout.strip() == "None None"


def test_every_recipe_option_is_checked_against_tool_help():
    from andarilho_worker.recipes import ARGV
    from andarilho_worker.toolcheck import HELP, options

    unchecked = [name for name, template in ARGV.items() if options(template) and name not in HELP]
    assert unchecked == []


def test_toolcheck_reports_options_the_tool_does_not_accept():
    from andarilho_worker.recipes import ARGV
    from andarilho_worker.toolcheck import missing_options, options

    every_option = "\n".join(opt for template in ARGV.values() for opt in options(template))
    assert missing_options(lambda argv: every_option) == {}
    # COLMAP 3.12 names the GPU switch SiftExtraction.use_gpu; a renamed option is caught,
    # and a longer option sharing the prefix does not hide it.
    stale = every_option.replace("--SiftExtraction.use_gpu", "--SiftExtraction.use_gpu_index")
    missing = missing_options(lambda argv: stale)
    assert missing == {"colmap_feature_extractor": ["--SiftExtraction.use_gpu"]}
