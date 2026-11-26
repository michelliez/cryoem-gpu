"""GPU smoke test (spec §5.3, M4.5): time the fused pipeline's `compute` tier on ~100 particles
with perf_counter and with CUDA events, and print both. Prints only: it writes no files and
never touches results/.

    python scripts/smoke_gpu.py --star data/subset.star
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from cryoem_gpu import config, pipeline
from cryoem_gpu.backend import Gpu
from cryoem_gpu.ctf import CtfParams
from cryoem_gpu.io import preload
from cryoem_gpu.star import read_star


def load(star: Path, n: int):
    """The first n particles of star (image paths resolve against its directory) and their CTF."""
    # preload orders by (file, index); sort the same way so the columns line up with the images.
    particles = sorted(read_star(star), key=lambda p: (p.file, p.index))
    images = preload(particles, star.parent, n)
    columns = CtfParams(
        *(np.array([getattr(p, f) for p in particles[:n]], np.float32) for f in CtfParams._fields)
    )
    return images, pipeline.StackParams(ctf=columns, pixel_size=particles[0].pixel_size)


def time_compute(backend, cuda, run, trials):
    """Seconds per trial of run(), from perf_counter and from CUDA events, after one warm-up.

    Both clocks bracket the same work; the device is synchronized before starting and before
    stopping the perf_counter clock (spec §5.3).
    """
    run()
    backend.sync()
    perf, events = [], []
    for _ in range(trials):
        start, end = cuda.Event(), cuda.Event()
        backend.sync()
        t0 = time.perf_counter()
        start.record()
        run()
        end.record()
        backend.sync()
        perf.append(time.perf_counter() - t0)
        events.append(cuda.get_elapsed_time(start, end) / 1e3)
    return perf, events


def _summary(label, seconds):
    ms = [s * 1e3 for s in seconds]
    return f"  {label:<13} median {statistics.median(ms):9.3f} ms  min {min(ms):9.3f}  max {max(ms):9.3f}"


def main(argv=None, backend=None, cuda=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--star", type=Path, required=True, help="subset.star from download_subset.py")
    parser.add_argument("--n", type=int, default=100, help="particles (default 100)")
    parser.add_argument("--trials", type=int, default=20, help="timed trials (default 20)")
    args = parser.parse_args(argv)

    if backend is None:
        try:
            backend = Gpu()
        except (ImportError, RuntimeError) as e:
            sys.exit(f"smoke_gpu.py needs CuPy and a CUDA GPU: {e}")
    cuda = cuda or backend.xp.cuda

    images, params = load(args.star, args.n)
    # `compute` tier: every chunk and its CTF columns are already on the device.
    chunks = [
        (
            backend.to_device(images[s : s + config.CHUNK_SIZE]),
            CtfParams(*(backend.to_device(c[s : s + config.CHUNK_SIZE]) for c in params.ctf)),
        )
        for s in range(0, len(images), config.CHUNK_SIZE)
    ]

    def run():
        for x, ctf_params in chunks:
            pipeline.process_chunk(backend, x, ctf_params, params.pixel_size)

    perf, events = time_compute(backend, cuda, run, args.trials)
    h, w = images.shape[1:]
    print(
        f"{backend.name} compute, fused pipeline, {len(images)} particles of {h}x{w}, "
        f"chunk {config.CHUNK_SIZE}, {args.trials} trials after 1 warm-up:"
    )
    print(_summary("perf_counter", perf))
    print(_summary("CUDA events", events))
    ratio = statistics.median(perf) / statistics.median(events)
    print(f"  perf_counter / CUDA events (medians) = {ratio:.3f}")


if __name__ == "__main__":
    main()
