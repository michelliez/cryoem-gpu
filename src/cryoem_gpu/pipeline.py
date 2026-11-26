"""Per-op functions, the fused per-chunk pipeline, and the chunked stack driver (spec §3.5, §4).

normalize → F = rfft2(x) → F ·= L · sign(CTF) → irfft2(F), one FFT pair per chunk, "backward" norm.
"""

from typing import NamedTuple

import numpy as np

from cryoem_gpu import config, filters
from cryoem_gpu.ctf import CtfParams, ctf, flip_sign

_AXES = (-2, -1)


class StackParams(NamedTuple):
    """What process_stack needs besides the images: per-particle CTF columns (host, (N,)) and Å/px."""

    ctf: CtfParams
    pixel_size: float


def normalize(backend, x):
    """Background-normalize each particle of a (B, H, W) chunk."""
    return filters.normalize(backend.xp, x)


def rfft2(backend, x):
    """Batched rfft over the last two axes."""
    return backend.rfftn(x, axes=_AXES)


def irfft2(backend, f, shape):
    """Batched inverse rfft back to real-space image `shape`."""
    return backend.irfftn(f, s=shape, axes=_AXES)


def _flip_factor(backend, shape, pixel_size, ctf_params):
    xp = backend.xp
    return flip_sign(xp, ctf(xp, filters.frequency_grid(xp, shape, pixel_size), ctf_params))


def lowpass_multiply(backend, f, shape, pixel_size):
    """F · L for real-space image `shape`."""
    return f * filters.lowpass(backend.xp, shape, pixel_size)


def ctf_flip(backend, f, shape, pixel_size, ctf_params):
    """F · sign(CTF), with ctf_params as (B,) columns on the backend."""
    return f * _flip_factor(backend, shape, pixel_size, ctf_params)


def process_chunk(backend, x, ctf_params, pixel_size):
    """The fused pipeline on one (B, H, W) chunk already on the backend."""
    shape = x.shape[-2:]
    f = rfft2(backend, normalize(backend, x))
    f *= filters.lowpass(backend.xp, shape, pixel_size) * _flip_factor(
        backend, shape, pixel_size, ctf_params
    )
    return irfft2(backend, f, shape)


def process_stack(host_stack, params, backend, chunk_size=config.CHUNK_SIZE):
    """Run process_chunk over host_stack in chunks of chunk_size, into a new host array.

    Each chunk and its CTF columns go in through backend.to_device and come back through
    backend.to_host into the chunk's slice of the output. backend.map decides whether chunks
    run in parallel.
    """
    out = np.empty_like(host_stack)
    # Same dtype as the images, so device backends reuse one staging buffer for both.
    columns = CtfParams(*(np.asarray(col, dtype=host_stack.dtype) for col in params.ctf))

    def run(start):
        rows = slice(start, start + chunk_size)
        x = backend.to_device(host_stack[rows])
        ctf_params = CtfParams(*(backend.to_device(col[rows]) for col in columns))
        backend.to_host(process_chunk(backend, x, ctf_params, params.pixel_size), out[rows])

    backend.map(run, range(0, len(host_stack), chunk_size))
    backend.sync()
    return out
