"""Frequency grids, the raised-cosine low-pass, and background normalization (spec §3.1–3.3).

Every function takes the backend's array module `xp`, and the cached arrays live on it.
Cached arrays are shared between callers, so never modify them in place.
"""

from functools import lru_cache
from typing import NamedTuple

from cryoem_gpu import config


class FrequencyGrid(NamedTuple):
    """Spatial frequency of each rfft bin: s² in 1/Å², θ = atan2(ky, kx) over the last two axes."""

    s2: object
    theta: object


def _axis_freqs(xp, shape, d):
    """Per-axis frequencies shaped to broadcast over the rfft output: fftfreq, then rfftfreq last."""
    freqs = [xp.fft.fftfreq(n, d=d) for n in shape[:-1]] + [xp.fft.rfftfreq(shape[-1], d=d)]
    ndim = len(shape)
    return [f.reshape((-1,) + (1,) * (ndim - 1 - i)) for i, f in enumerate(freqs)]


@lru_cache(maxsize=None)
def frequency_grid(xp, shape, pixel_size):
    """The cached frequency grid for real-space `shape` sampled at `pixel_size` Å."""
    freqs = _axis_freqs(xp, shape, pixel_size)
    s2 = sum(f**2 for f in freqs)
    theta = xp.arctan2(freqs[-2], freqs[-1])
    return FrequencyGrid(s2.astype(xp.float32), xp.broadcast_to(theta, s2.shape).astype(xp.float32))


@lru_cache(maxsize=None)
def lowpass(xp, shape, pixel_size, cutoff_a=config.LOWPASS_CUTOFF_A, edge_px=config.LOWPASS_EDGE_PX):
    """Raised-cosine low-pass over the rfft bins of a cubic `shape`, in Fourier-pixel radius.

    1 up to r_c = box·pixel_size/cutoff_a, a half-cosine fall over edge_px pixels, then 0.
    """
    box = shape[0]
    if any(n != box for n in shape):
        raise ValueError(f"lowpass needs equal dimensions, got {shape}")
    # d=1/n gives integer Fourier indices, so r is exact and independent of pixel size.
    r = xp.sqrt(sum(f**2 for f in _axis_freqs(xp, shape, 1.0 / box)))
    r_c = box * pixel_size / cutoff_a
    t = xp.clip((r - r_c) / edge_px, 0.0, 1.0)
    return (0.5 * (1.0 + xp.cos(xp.pi * t))).astype(xp.float32)


@lru_cache(maxsize=None)
def _background_mask(xp, shape, radius_px):
    yy = xp.arange(shape[0]).reshape(-1, 1) - shape[0] // 2
    xx = xp.arange(shape[1]).reshape(1, -1) - shape[1] // 2
    return yy**2 + xx**2 > radius_px**2


def normalize(xp, x, radius_px=config.BG_RADIUS_PX):
    """RELION-style: per particle, subtract the background mean and divide by its std.

    The background is the pixels farther than radius_px from the image center (H//2, W//2).
    The statistics accumulate in float64 (float32 drifts ~1e-4 on offset data); the result is x's dtype.
    """
    bg = x[:, _background_mask(xp, x.shape[-2:], radius_px)]
    mean = bg.mean(axis=1, dtype=xp.float64).astype(x.dtype).reshape(-1, 1, 1)
    std = bg.std(axis=1, dtype=xp.float64).astype(x.dtype).reshape(-1, 1, 1)
    return (x - mean) / std
