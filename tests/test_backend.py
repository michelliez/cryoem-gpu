import os
import threading

import numpy as np
import pytest

from cryoem_gpu import config
from cryoem_gpu.backend import CpuSerial, CpuThreaded, FakeDevice, usable_cores


def _rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture
def stack():
    return np.random.default_rng(0).standard_normal((6, 360, 360), dtype=np.float32)


def _round_trip(backend, host):
    dev = backend.to_device(host)
    spectrum = backend.rfftn(dev, axes=(-2, -1))
    image = backend.irfftn(spectrum, s=host.shape[-2:], axes=(-2, -1))
    out = np.empty_like(host)
    backend.to_host(image, out)
    backend.sync()
    return spectrum, out


def test_rfft2_irfft2_round_trip_within_tolerance(stack):
    spectrum, out = _round_trip(CpuSerial(), stack)
    assert spectrum.shape == (6, 360, 181)
    assert spectrum.dtype == np.complex64
    assert out.dtype == np.float32
    assert _rel_l2(out, stack) < config.TOLERANCES["roundtrip"]


def test_fake_device_matches_cpu_serial(stack):
    expected_spectrum, expected = _round_trip(CpuSerial(), stack)
    spectrum, out = _round_trip(FakeDevice(), stack)
    assert _rel_l2(spectrum, expected_spectrum) < config.TOLERANCES["rfft2"]
    assert _rel_l2(out, expected) < config.TOLERANCES["irfft2"]


def test_fake_device_transfers_copy_instead_of_aliasing(stack):
    backend = FakeDevice()
    host = stack.copy()
    dev = backend.to_device(host)
    host[...] = 0
    out = np.zeros_like(host)
    backend.to_host(dev, out)
    dev[...] = 1
    np.testing.assert_array_equal(out, stack)


def test_usable_cores_is_a_positive_int():
    n = usable_cores()
    assert isinstance(n, int)
    assert n >= 1


def test_usable_cores_prefers_the_affinity_mask(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 3}, raising=False)
    assert usable_cores() == 2


@pytest.mark.parametrize("backend_cls", [CpuSerial, CpuThreaded, FakeDevice])
def test_chunked_map_matches_whole_stack_cpu_serial(backend_cls, stack):
    _, expected = _round_trip(CpuSerial(), stack)
    backend = backend_cls()
    out = np.empty_like(stack)

    def process(start):
        _, out[start : start + 4] = _round_trip(backend, stack[start : start + 4])

    backend.map(process, range(0, len(stack), 4))
    assert _rel_l2(out, expected) < config.TOLERANCES["irfft2"]


def test_cpu_threaded_map_runs_on_pool_threads():
    backend = CpuThreaded()
    names = backend.map(lambda _: threading.current_thread().name, range(4))
    assert threading.current_thread().name not in names


def test_fake_device_staging_handles_a_smaller_tail_chunk(stack):
    backend = FakeDevice()
    for chunk in (stack[:4], stack[4:], stack[:4]):
        out = np.empty_like(chunk)
        backend.to_host(backend.to_device(chunk), out)
        np.testing.assert_array_equal(out, chunk)
