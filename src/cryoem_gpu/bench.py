"""Benchmark bookkeeping: the environment fingerprint and the git_sha of the installed code."""

import importlib.metadata
import json
import os
import platform
import subprocess
import warnings
from pathlib import Path

from cryoem_gpu.backend import gpu_info, usable_cores

DIST_NAME = "cryoem-gpu"


def _cpu_model():
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version(import_name):
    """Installed version of the distribution providing import_name (e.g. cupy-cuda12x), or None."""
    for dist in importlib.metadata.packages_distributions().get(import_name, []):
        return importlib.metadata.Distribution.from_name(dist).version
    return None


def fingerprint() -> dict:
    """The environment fields of spec §5.4, in column order."""
    return {
        "cpu_model": _cpu_model(),
        "n_cores_usable": usable_cores(),
        **gpu_info(),
        "python": platform.python_version(),
        **{name: _version(name) for name in ("numpy", "scipy", "cupy", "mrcfile")},
    }


def write_environment(results_dir) -> dict:
    """Write the fingerprint plus git_sha and its source to results_dir/environment.json."""
    sha, source = resolve_git_sha()
    env = {**fingerprint(), "git_sha": sha, "git_sha_source": source}
    path = Path(results_dir) / "environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2) + "\n")
    return env


def _direct_url_commit():
    try:
        text = importlib.metadata.distribution(DIST_NAME).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def _git_head():
    """HEAD of the checkout holding this source file, for local editable installs."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_git_sha() -> tuple[str, str]:
    """(sha, source) for the installed package code, trying the sources in spec §5.4 order."""
    sha = _direct_url_commit()
    if sha:
        return sha, "direct_url"
    sha = os.environ.get("CRYOEM_GPU_COMMIT")
    if sha:
        return sha, "env"
    sha = _git_head()
    if sha:
        return sha, "git"
    warnings.warn(
        "git_sha unknown: no pip VCS metadata, no CRYOEM_GPU_COMMIT, and git rev-parse failed",
        stacklevel=2,
    )
    return "unknown", "unknown"
