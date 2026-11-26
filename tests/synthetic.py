"""Synthetic particle stacks and STAR-like CTF columns shared by the pipeline and agreement tests."""

import numpy as np

from cryoem_gpu import pipeline
from cryoem_gpu.ctf import CtfParams

PIXEL_SIZE = 1.34
BOX = 360


def stack(n):
    return np.random.default_rng(0).standard_normal((n, BOX, BOX), dtype=np.float32) + 50


def params(n):
    """Host-side STAR columns with distinct defocus per particle, so a misaligned chunk shows."""
    rng = np.random.default_rng(1)
    u = rng.uniform(15000, 32000, n)
    columns = dict(
        defocus_u=u,
        defocus_v=u - rng.uniform(0, 2000, n),
        defocus_angle=rng.uniform(0, 180, n),
        voltage=np.full(n, 300.0),
        cs=np.full(n, 2.7),
        amp_contrast=np.full(n, 0.1),
    )
    ctf = CtfParams(**{k: v.astype(np.float32) for k, v in columns.items()})
    return pipeline.StackParams(ctf=ctf, pixel_size=PIXEL_SIZE)
