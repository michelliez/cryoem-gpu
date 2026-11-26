import gzip
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cryoem_gpu.star import read_star

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_subset.py"
IMAGE_BYTES = 360 * 360 * 4

HEADER = """
data_

loop_
_rlnVoltage #1
_rlnDefocusU #2
_rlnDefocusV #3
_rlnDefocusAngle #4
_rlnSphericalAberration #5
_rlnDetectorPixelSize #6
_rlnCtfFigureOfMerit #7
_rlnMagnification #8
_rlnAmplitudeContrast #9
_rlnImageName #10
"""
# Stacks per file number, in particles. File 003 must not reach subset.star at --n-files 2.
COUNTS = {1: 2, 2: 3, 3: 1}


def stack_rel(i: int) -> str:
    return f"Particles/MRC_0601/{i:03d}_particles_shiny_nb50_new.mrcs"


def stack_size(n: int) -> int:
    return 1024 + n * IMAGE_BYTES


@pytest.fixture
def server(tmp_path):
    """Serve a fake EMPIAR tree: data/Particles/... plus a gzipped volume."""
    root = tmp_path / "remote"
    particles = root / "data" / "Particles"
    (particles / "MRC_0601").mkdir(parents=True)
    rows = []
    for i, n in COUNTS.items():
        (root / "data" / stack_rel(i)).write_bytes(b"\0" * stack_size(n))
        rows += [
            f"300.0 20000.0 19000.0 40.0 2.0 14.0 0.13 104478.0 0.1 {k:06d}@{stack_rel(i)}"
            for k in range(1, n + 1)
        ]
    (particles / "shiny_2sets.star").write_text(HEADER + "\n".join(rows) + "\n")
    (root / "emd_2660.map.gz").write_bytes(gzip.compress(b"volume-bytes"))

    requests: list[str] = []

    class Recorder(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            super().do_GET()

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Recorder, directory=str(root)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    yield base, requests
    httpd.shutdown()


def run(base: str, out: Path, *extra: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out),
            "--base-url",
            f"{base}/data/",
            "--volume-url",
            f"{base}/emd_2660.map.gz",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def stack_requests(requests: list[str]) -> list[str]:
    return [r for r in requests if r.endswith(".mrcs")]


def test_downloads_requested_files_and_writes_subset_star(server, tmp_path):
    base, requests = server
    out = tmp_path / "out"

    run(base, out, "--n-files", "2")

    for i in (1, 2):
        assert (out / stack_rel(i)).stat().st_size == stack_size(COUNTS[i])
    assert not (out / stack_rel(3)).exists()
    subset = read_star(out / "subset.star")
    assert [(p.file, p.index) for p in subset] == [
        (stack_rel(1), 1),
        (stack_rel(1), 2),
        (stack_rel(2), 1),
        (stack_rel(2), 2),
        (stack_rel(2), 3),
    ]
    assert len(stack_requests(requests)) == 2


def test_rerun_downloads_nothing(server, tmp_path):
    base, requests = server
    out = tmp_path / "out"
    run(base, out, "--n-files", "2")
    requests.clear()

    run(base, out, "--n-files", "2")

    assert requests == []


def test_truncated_stack_is_redownloaded(server, tmp_path):
    base, requests = server
    out = tmp_path / "out"
    run(base, out, "--n-files", "2")
    with open(out / stack_rel(2), "r+b") as f:
        f.truncate(stack_size(COUNTS[2]) - 100)
    requests.clear()

    run(base, out, "--n-files", "2")

    assert stack_requests(requests) == [f"/data/{stack_rel(2)}"]
    assert (out / stack_rel(2)).stat().st_size == stack_size(COUNTS[2])


def test_reports_throughput_and_free_disk(server, tmp_path):
    base, _ = server

    stdout = run(base, tmp_path / "out", "--n-files", "2")

    assert "MB/s" in stdout
    assert "free" in stdout.lower()


def test_volume_flag_fetches_and_decompresses_the_map(server, tmp_path):
    base, requests = server
    out = tmp_path / "out"

    run(base, out, "--n-files", "1")
    assert not (out / "emd_2660.map").exists()

    run(base, out, "--n-files", "1", "--volume")
    assert (out / "emd_2660.map").read_bytes() == b"volume-bytes"
    requests.clear()

    run(base, out, "--n-files", "1", "--volume")
    assert requests == []


def test_file_missing_from_star_is_an_error(server, tmp_path):
    base, _ = server

    with pytest.raises(subprocess.CalledProcessError) as e:
        run(base, tmp_path / "out", "--n-files", "4")

    assert "004" in e.value.stderr
