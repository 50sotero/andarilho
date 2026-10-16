from __future__ import annotations

import subprocess

from andarilho_worker import main as main_module


def _stub_tools(monkeypatch, present: set[str], *, smi_fails: bool = False) -> None:
    """Pretend only ``present`` executables are on PATH; ``nvidia-smi -L`` can fail."""

    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    def run(argv, **kwargs):
        if smi_fails:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(main_module.shutil, "which", which)
    monkeypatch.setattr(main_module.subprocess, "run", run)


def test_gpu_readiness_needs_nvidia_smi_and_colmap_only(monkeypatch):
    # The worker no longer runs any training tool, so their absence blocks nothing.
    _stub_tools(monkeypatch, {"nvidia-smi", "colmap"})
    assert main_module.capability_problems("gpu") == []


def test_gpu_worker_refuses_to_start_without_colmap(monkeypatch):
    _stub_tools(monkeypatch, {"nvidia-smi"})
    assert main_module.capability_problems("gpu") == ["colmap not found"]


def test_gpu_worker_refuses_to_start_when_nvidia_smi_fails(monkeypatch):
    _stub_tools(monkeypatch, {"nvidia-smi", "colmap"}, smi_fails=True)
    assert main_module.capability_problems("gpu") == ["nvidia-smi failed"]


def test_cpu_capability_needs_no_gpu_tools(monkeypatch):
    _stub_tools(monkeypatch, set())
    assert main_module.capability_problems("cpu") == []
