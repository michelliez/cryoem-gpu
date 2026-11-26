import numpy as np
import pytest

from cryoem_gpu import config
from cryoem_gpu.backend import CpuSerial, FakeDevice
from cryoem_gpu.ctf import CtfParams, ctf, flip_sign, wavelength
from cryoem_gpu.filters import FrequencyGrid, frequency_grid

PIXEL_SIZE = 1.34
BOX = 64
BACKENDS = [CpuSerial(), FakeDevice()]


def _params(xp, **overrides):
    """Three particles with EMPIAR-10028-like values; overrides replace whole columns."""
    columns = dict(
        defocus_u=[18000.0, 25000.0, 31000.0],
        defocus_v=[17000.0, 24500.0, 29000.0],
        defocus_angle=[10.0, 45.0, 120.0],
        voltage=[300.0] * 3,
        cs=[2.7] * 3,
        amp_contrast=[0.1, 0.07, 0.15],
    )
    columns.update(overrides)
    return CtfParams(**{k: xp.asarray(v, dtype=xp.float32) for k, v in columns.items()})


def test_wavelength_at_300_kv():
    assert wavelength(300.0) == pytest.approx(0.019687, abs=config.TOLERANCES["wavelength"])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_ctf_at_zero_frequency_is_amplitude_contrast(backend):
    xp = backend.xp
    p = _params(xp)
    c = ctf(xp, frequency_grid(xp, (BOX, BOX), PIXEL_SIZE), p)
    assert c.shape == (3, BOX, BOX // 2 + 1)
    assert c.dtype == np.float32
    np.testing.assert_allclose(c[:, 0, 0], p.amp_contrast, atol=config.TOLERANCES["ctf_closed_form"])


def test_ctf_rises_from_amplitude_contrast_before_first_zero():
    # RELION's + sin χ form: under positive (under-)focus the CTF climbs from A toward 1
    # at low s. The cryoDRGN − form would fall from A through 0 instead.
    s2 = np.linspace(0.0, 1e-3, 101, dtype=np.float32)
    params = _params(np, defocus_u=[20000.0], defocus_v=[20000.0], defocus_angle=[0.0], voltage=[300.0], cs=[2.7], amp_contrast=[0.1])
    c = ctf(np, FrequencyGrid(s2, np.zeros_like(s2)), params)[0]
    assert np.all(np.diff(c) > 0)
    assert c[-1] > 0.9


def test_non_astigmatic_sign_changes_fall_at_analytic_chi_zeros():
    defocus, cs_a = 20000.0, 2.7e7
    s2_max = (1 / (2 * PIXEL_SIZE)) ** 2
    s2 = np.linspace(0.0, s2_max, 200_001, dtype=np.float32)
    grid = FrequencyGrid(s2, np.zeros_like(s2))
    params = _params(
        np, defocus_u=[defocus], defocus_v=[defocus], defocus_angle=[0.0], voltage=[300.0], cs=[2.7], amp_contrast=[0.0]
    )
    c = ctf(np, grid, params)[0]

    # A = 0 gives CTF = sin χ, zero where χ = nπ:  λΔf·s² − (Cs·λ³/2)·s⁴ = n.
    lam = wavelength(300.0)
    n = np.arange(1, 200)
    zeros = (defocus - np.sqrt(defocus**2 - 2 * n * cs_a * lam)) / (cs_a * lam**2)
    zeros = zeros[zeros < s2_max]

    changes = np.flatnonzero(np.signbit(c[1:]) != np.signbit(c[:-1]))
    assert len(changes) == len(zeros) > 20
    step = s2[1] - s2[0]
    np.testing.assert_allclose(s2[changes], zeros, atol=config.TOLERANCES["ctf_zero_steps"] * step)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_swapping_u_and_v_with_angle_plus_90_is_invariant(backend):
    xp = backend.xp
    grid = frequency_grid(xp, (BOX, BOX), PIXEL_SIZE)
    p = _params(xp)
    swapped = p._replace(defocus_u=p.defocus_v, defocus_v=p.defocus_u)
    rotated = swapped._replace(defocus_angle=p.defocus_angle + 90)

    np.testing.assert_allclose(ctf(xp, grid, rotated), ctf(xp, grid, p), atol=config.TOLERANCES["ctf_closed_form"])
    assert np.abs(ctf(xp, grid, swapped) - ctf(xp, grid, p)).max() > 0.5


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_each_particle_uses_its_own_parameters(backend):
    xp = backend.xp
    grid = frequency_grid(xp, (BOX, BOX), PIXEL_SIZE)
    p = _params(xp)
    batch = ctf(xp, grid, p)
    for i in range(3):
        one = CtfParams(*(col[i : i + 1] for col in p))
        np.testing.assert_array_equal(batch[i], ctf(xp, grid, one)[0])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_flip_sign_is_plus_or_minus_one_with_sign_of_zero_positive(backend):
    xp = backend.xp
    c = xp.asarray([-0.7, -1e-30, -0.0, 0.0, 1e-30, 0.3], dtype=xp.float32)
    s = flip_sign(xp, c)
    assert s.dtype == np.float32
    np.testing.assert_array_equal(s, [-1, -1, 1, 1, 1, 1])
