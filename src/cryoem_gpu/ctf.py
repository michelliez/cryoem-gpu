"""CTF in the RELION convention (spec §3.4): no B-factor, no phase shift.

Parameters come in STAR units (defocus Å, angle degrees, voltage kV, Cs mm, amplitude contrast)
and are converted here.
"""

import math
from typing import NamedTuple


class CtfParams(NamedTuple):
    """Per-particle CTF parameters in STAR units, each a (B,) array on the active backend."""

    defocus_u: object
    defocus_v: object
    defocus_angle: object
    voltage: object
    cs: object
    amp_contrast: object


def wavelength(voltage_kv):
    """Relativistic electron wavelength in Å."""
    v = voltage_kv * 1e3
    return 12.2643247 / (v * (1.0 + 0.978466e-6 * v)) ** 0.5


def ctf(xp, grid, params):
    """CTF = √(1−A²)·sin χ + A·cos χ for each particle over `grid`, shape (B, *grid.s2.shape)."""
    per_particle = (-1,) + (1,) * grid.s2.ndim
    p = CtfParams(*(col.reshape(per_particle) for col in params))
    u, v = p.defocus_u, p.defocus_v
    lam = wavelength(p.voltage)
    defocus = (u + v) / 2 + (u - v) / 2 * xp.cos(2 * (grid.theta - xp.deg2rad(p.defocus_angle)))
    cs_angstrom = p.cs * 1e7
    chi = math.pi * lam * defocus * grid.s2 - (math.pi / 2) * cs_angstrom * lam**3 * grid.s2**2
    amp = p.amp_contrast
    return xp.sqrt(1 - amp**2) * xp.sin(chi) + amp * xp.cos(chi)


def flip_sign(xp, c):
    """Phase-flip factor sign(c) as ±1 in c's dtype, with sign(0) = +1."""
    return xp.where(c >= 0, 1, -1).astype(c.dtype)
