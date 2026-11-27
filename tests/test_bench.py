import json
import subprocess
import sys
import types
from pathlib import Path

import numpy
import pytest
import scipy

from cryoem_gpu import bench
from cryoem_gpu.backend import usable_cores

SHA = "0123456789abcdef0123456789abcdef01234567"


class _Dist:
    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return None if self._direct_url is None else json.dumps(self._direct_url)


@pytest.fixture
def installed(monkeypatch):
    """Set what pip's direct_url.json says about the installed package (None: no file)."""
    monkeypatch.delenv("CRYOEM_GPU_COMMIT", raising=False)

    def install(direct_url):
        monkeypatch.setattr(bench.importlib.metadata, "distribution", lambda name: _Dist(direct_url))

    return install


def test_sha_from_pip_vcs_install(installed, monkeypatch):
    installed({"url": "https://github.com/michelliez/cryoem-gpu", "vcs_info": {"vcs": "git", "commit_id": SHA}})
    monkeypatch.setenv("CRYOEM_GPU_COMMIT", "f" * 40)
    assert bench.resolve_git_sha() == (SHA, "direct_url")


@pytest.mark.parametrize(
    "direct_url",
    [None, {"url": "file:///src/cryoem-gpu", "dir_info": {"editable": True}}],
    ids=["no-direct-url", "editable"],
)
def test_sha_from_env_when_pip_has_no_commit(installed, monkeypatch, direct_url):
    installed(direct_url)
    monkeypatch.setenv("CRYOEM_GPU_COMMIT", SHA)
    assert bench.resolve_git_sha() == (SHA, "env")


def test_sha_from_git_for_local_checkout(installed, monkeypatch, tmp_path):
    installed(None)
    monkeypatch.chdir(tmp_path)  # resolves against the package source, not the cwd
    repo = Path(__file__).resolve().parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    assert bench.resolve_git_sha() == (head.stdout.strip(), "git")


def test_environment_json_records_fingerprint_and_sha_source(installed, monkeypatch, tmp_path):
    installed(None)
    monkeypatch.setenv("CRYOEM_GPU_COMMIT", SHA)
    bench.write_environment(tmp_path)
    env = json.loads((tmp_path / "environment.json").read_text())
    assert set(env) == {
        "cpu_model", "n_cores_usable", "gpu_name", "driver", "cuda_runtime",
        "python", "numpy", "scipy", "cupy", "mrcfile", "git_sha", "git_sha_source",
    }  # fmt: skip
    assert (env["git_sha"], env["git_sha_source"]) == (SHA, "env")
    assert env["n_cores_usable"] == usable_cores()
    assert (env["numpy"], env["scipy"]) == (numpy.__version__, scipy.__version__)
    assert env["cpu_model"]


def test_n_cores_usable_comes_from_usable_cores(monkeypatch):
    monkeypatch.setattr(bench, "usable_cores", lambda: 3)
    assert bench.fingerprint()["n_cores_usable"] == 3


def test_sha_unknown_warns_when_no_source(installed, monkeypatch, tmp_path):
    installed(None)
    monkeypatch.setenv("PATH", str(tmp_path))  # no git executable
    with pytest.warns(UserWarning, match="git_sha"):
        assert bench.resolve_git_sha() == ("unknown", "unknown")


def test_fingerprint_gpu_fields_are_none_without_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    env = bench.fingerprint()
    assert (env["gpu_name"], env["driver"], env["cuda_runtime"]) == (None, None, None)


def test_fingerprint_gpu_fields_with_cuda_device(monkeypatch, tmp_path):
    runtime = types.SimpleNamespace(
        getDeviceProperties=lambda i: {"name": b"Tesla T4"}, runtimeGetVersion=lambda: 12020
    )
    cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=runtime))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 535.104.05\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    env = bench.fingerprint()
    assert (env["gpu_name"], env["driver"], env["cuda_runtime"]) == ("Tesla T4", "535.104.05", "12.2")
