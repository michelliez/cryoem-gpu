import numpy as np
import pytest

from cryoem_gpu import config
from cryoem_gpu.backend import CpuSerial, FakeDevice
from cryoem_gpu.filters import frequency_grid, lowpass, normalize

PIXEL_SIZE = 1.34
BOX = 360


def test_frequency_grid_matches_spec_and_is_float32():
    grid = frequency_grid(np, (BOX, BOX), PIXEL_SIZE)
    kx = np.fft.rfftfreq(BOX, d=PIXEL_SIZE)
    ky = np.fft.fftfreq(BOX, d=PIXEL_SIZE)
    assert grid.s2.shape == grid.theta.shape == (BOX, BOX // 2 + 1)
    assert grid.s2.dtype == grid.theta.dtype == np.float32
    np.testing.assert_allclose(grid.s2, kx[None, :] ** 2 + ky[:, None] ** 2, rtol=config.TOLERANCES["freq_grid"])
    np.testing.assert_allclose(grid.theta, np.arctan2(ky[:, None], kx[None, :]), atol=config.TOLERANCES["freq_grid"])


def test_frequency_grid_is_cached_per_shape_and_pixel_size():
    a = frequency_grid(np, (BOX, BOX), PIXEL_SIZE)
    assert frequency_grid(np, (BOX, BOX), PIXEL_SIZE) is a
    assert frequency_grid(np, (BOX, BOX), 1.0) is not a
    assert frequency_grid(np, (64, 64), PIXEL_SIZE) is not a


def _radius_px(shape):
    """Fourier-pixel radius of each rfft bin, computed independently of filters.py."""
    axes = [np.fft.fftfreq(n) * n for n in shape[:-1]] + [np.fft.rfftfreq(shape[-1]) * shape[-1]]
    return np.sqrt(sum(a**2 for a in np.meshgrid(*axes, indexing="ij")))


@pytest.mark.parametrize("shape", [(BOX, BOX), (64, 64, 64)])
def test_lowpass_profile(shape):
    box = shape[0]
    L = lowpass(np, shape, PIXEL_SIZE)
    r = _radius_px(shape)
    r_c = box * PIXEL_SIZE / config.LOWPASS_CUTOFF_A
    w = config.LOWPASS_EDGE_PX

    assert L.shape == r.shape
    assert L.dtype == np.float32
    assert np.all(L[r <= r_c] == 1)
    assert np.all(L[r >= r_c + w] == 0)
    edge = (r > r_c) & (r < r_c + w)
    assert edge.any()
    assert np.all((L[edge] > 0) & (L[edge] < 1))
    order = np.argsort(r.ravel(), kind="stable")
    assert np.all(np.diff(L.ravel()[order]) <= 0)


def test_lowpass_cutoff_and_width_are_configurable():
    L = lowpass(np, (BOX, BOX), PIXEL_SIZE, cutoff_a=20.0, edge_px=2.0)
    r = _radius_px((BOX, BOX))
    r_c = BOX * PIXEL_SIZE / 20.0
    assert np.all(L[r <= r_c] == 1)
    assert np.all(L[r >= r_c + 2.0] == 0)
    assert L[(r > r_c) & (r < r_c + 2.0)].size > 0


def test_lowpass_rejects_non_cubic_shape():
    with pytest.raises(ValueError):
        lowpass(np, (BOX, 64), PIXEL_SIZE)


def _background(shape, radius):
    yy, xx = np.indices(shape)
    return np.hypot(yy - shape[0] // 2, xx - shape[1] // 2) > radius


@pytest.mark.parametrize("backend", [CpuSerial(), FakeDevice()], ids=lambda b: b.name)
def test_normalize_gives_zero_mean_unit_std_background(backend):
    rng = np.random.default_rng(1)
    host = (rng.standard_normal((4, BOX, BOX)) * [[[3]], [[0.5]], [[7]], [[1]]] + [[[10]], [[-2]], [[0]], [[5]]]).astype(np.float32)
    out = backend.to_host(normalize(backend.xp, backend.to_device(host)), np.empty_like(host))
    bg = out[:, _background((BOX, BOX), config.BG_RADIUS_PX)]
    assert out.dtype == np.float32
    np.testing.assert_allclose(bg.mean(axis=1), 0, atol=config.TOLERANCES["background_stats"])
    np.testing.assert_allclose(bg.std(axis=1), 1, atol=config.TOLERANCES["background_stats"])


def test_normalize_radius_is_configurable():
    rng = np.random.default_rng(2)
    host = rng.standard_normal((2, BOX, BOX)).astype(np.float32)
    host[:, _background((BOX, BOX), 100)] += 50  # only the ring beyond r=100 is shifted
    out = normalize(np, host, radius_px=100)
    bg = out[:, _background((BOX, BOX), 100)]
    np.testing.assert_allclose(bg.mean(axis=1), 0, atol=config.TOLERANCES["background_stats"])
    np.testing.assert_allclose(bg.std(axis=1), 1, atol=config.TOLERANCES["background_stats"])


@pytest.mark.parametrize("backend", [CpuSerial(), FakeDevice()], ids=lambda b: b.name)
def test_dtypes_float32_complex64_through_filter_chain(backend):
    host = np.random.default_rng(3).standard_normal((2, BOX, BOX), dtype=np.float32)
    x = backend.to_device(host)
    n = normalize(backend.xp, x)
    F = backend.rfftn(n, axes=(-2, -1))
    F = F * lowpass(backend.xp, (BOX, BOX), PIXEL_SIZE)
    y = backend.irfftn(F, s=(BOX, BOX), axes=(-2, -1))
    out = backend.to_host(y, np.empty_like(host))
    assert [a.dtype for a in (x, n, F, y, out)] == [np.float32, np.float32, np.complex64, np.float32, np.float32]
