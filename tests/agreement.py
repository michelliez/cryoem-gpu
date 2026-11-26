"""Agreement metrics for comparing a backend's output against a reference (spec §6.1)."""

from typing import NamedTuple

import numpy as np


class Errors(NamedTuple):
    rel_l2: float
    rmse: float
    max_abs: float

    def __str__(self):
        return f"rel_l2={self.rel_l2:.3e} rmse={self.rmse:.3e} max_abs={self.max_abs:.3e}"


def errors(a, b):
    """a against reference b, computed in float64."""
    dtype = np.result_type(a, b, np.float64)
    a, b = np.asarray(a, dtype), np.asarray(b, dtype)
    diff = np.abs(a - b)
    return Errors(
        rel_l2=float(np.linalg.norm(diff) / np.linalg.norm(b)),
        rmse=float(np.sqrt(np.mean(diff**2))),
        max_abs=float(diff.max()),
    )


def sign_disagreement(a, b):
    """Fraction of elements whose phase-flip sign (sign(0) = +1) differs between a and b."""
    return float(np.mean((np.asarray(a) >= 0) != (np.asarray(b) >= 0)))
