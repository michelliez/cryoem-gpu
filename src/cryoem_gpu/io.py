"""Read particle stacks through mrcfile's memory map."""

from collections import Counter
from collections.abc import Iterator
from itertools import groupby
from os import PathLike
from pathlib import Path

import mrcfile
import numpy as np
from mrcfile.mrcmemmap import MrcMemmap

from cryoem_gpu.star import Particle


def _open(path: str | PathLike) -> MrcMemmap:
    return mrcfile.mmap(path, mode="r", permissive=True)


def iter_chunks(path: str | PathLike, chunk_size: int) -> Iterator[np.ndarray]:
    """Yield the stack as float32 arrays of up to chunk_size particles, copied out of the map."""
    with _open(path) as mrc:
        for start in range(0, mrc.data.shape[0], chunk_size):
            yield np.array(mrc.data[start : start + chunk_size], dtype=np.float32)


def preload(particles: list[Particle], root: str | PathLike, n: int) -> np.ndarray:
    """Read the first n particles, ordered by file then 1-based index, into one float32 array."""
    if not 1 <= n <= len(particles):
        raise ValueError(f"asked to preload {n} particles but {len(particles)} are available")
    chosen = sorted(particles, key=lambda p: (p.file, p.index))[:n]
    files = {p.file for p in chosen}
    check_counts([p for p in particles if p.file in files], root)

    out: np.ndarray | None = None
    row = 0
    for file, group in groupby(chosen, key=lambda p: p.file):
        indices = [p.index - 1 for p in group]
        with _open(Path(root) / file) as mrc:
            if out is None:
                out = np.empty((n, *mrc.data.shape[1:]), dtype=np.float32)
            out[row : row + len(indices)] = mrc.data[indices]
        row += len(indices)
    assert out is not None
    return out


def check_counts(particles: list[Particle], root: str | PathLike) -> None:
    """Raise if any file's STAR row count differs from its stack's header nz."""
    for file, rows in Counter(p.file for p in particles).items():
        with _open(Path(root) / file) as mrc:
            nz = int(mrc.header.nz)
        if rows != nz:
            raise ValueError(f"{file}: STAR has {rows} rows but the stack header has nz={nz}")
