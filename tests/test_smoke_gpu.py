import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest

from cryoem_gpu.backend import FakeDevice

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_gpu.py"
N_FILE = 5
STAR = """
data_

loop_
_rlnImageName #1
_rlnDefocusU #2
_rlnDefocusV #3
_rlnDefocusAngle #4
_rlnVoltage #5
_rlnSphericalAberration #6
_rlnAmplitudeContrast #7
_rlnDetectorPixelSize #8
_rlnMagnification #9
"""


def _load_script():
    spec = importlib.util.spec_from_file_location("smoke_gpu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEvent:
    """cupy.cuda.Event's record, timed on the host clock."""

    def record(self):
        self.t = time.perf_counter()


_FAKE_CUDA = SimpleNamespace(
    Event=_FakeEvent, get_elapsed_time=lambda start, end: (end.t - start.t) * 1e3
)


@pytest.fixture
def subset(tmp_path):
    """A download_subset-style layout: subset.star whose image paths resolve against its dir."""
    root = tmp_path / "data"
    stack = root / "Particles" / "001.mrcs"
    stack.parent.mkdir(parents=True)
    with mrcfile.new(stack) as mrc:
        mrc.set_data(np.random.default_rng(0).standard_normal((N_FILE, 360, 360), np.float32))
    rows = "".join(
        f"{i:06d}@Particles/001.mrcs {20000 + 100 * i} 19000 30 300 2.7 0.1 14 104477\n"
        for i in range(1, N_FILE + 1)
    )
    (root / "subset.star").write_text(STAR + rows)
    return root / "subset.star"


def _files(root):
    return {p: p.stat().st_mtime_ns for p in root.rglob("*")}


def test_prints_cuda_event_vs_perf_counter_compute_time_and_writes_nothing(
    subset, tmp_path, monkeypatch, capsys
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    before = _files(tmp_path)

    _load_script().main(
        ["--star", str(subset), "--n", "4", "--trials", "3"], backend=FakeDevice(), cuda=_FAKE_CUDA
    )

    out = capsys.readouterr().out
    assert "4 particles" in out
    assert "perf_counter" in out
    assert "CUDA events" in out
    assert _files(tmp_path) == before


@pytest.mark.skipif(importlib.util.find_spec("cupy") is not None, reason="CuPy is installed")
def test_without_cupy_exits_with_a_message(subset):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--star", str(subset)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "CuPy" in result.stderr
