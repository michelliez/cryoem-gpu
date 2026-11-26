"""Backend-vs-CpuSerial agreement for every op and the pipeline (spec §6.1, §6.2).

Each test runs on FakeDevice (CPU, so the suite itself is checked locally) and on Gpu
(`gpu` marker, skipped without CuPy or a CUDA device). Tolerances come from config only.
"""

import numpy as np
import pytest

import synthetic
from agreement import errors, sign_disagreement
from cryoem_gpu import config, pipeline
from cryoem_gpu.backend import CpuSerial, FakeDevice, Gpu
from cryoem_gpu.ctf import CtfParams, ctf
from cryoem_gpu.filters import frequency_grid


def test_errors_match_hand_computed_values():
    e = errors(np.array([1.0, 2.0, 2.0, 4.0]), np.array([1.0, 2.0, 2.0, 2.0]))
    # ‖a−b‖ = 2, ‖b‖ = √13, RMSE = √(4/4) = 1, max |a−b| = 2.
    assert e.rel_l2 == pytest.approx(2 / 13**0.5)
    assert e.rmse == pytest.approx(1.0)
    assert e.max_abs == pytest.approx(2.0)


def test_sign_disagreement_counts_pixels_whose_flip_sign_differs():
    # sign(0) = +1, so 0 vs 1e-9 agrees and 0 vs -1e-9 does not.
    a = np.array([0.5, -0.5, 0.0, 0.0, 3.0])
    b = np.array([0.4, 0.5, 1e-9, -1e-9, 2.0])
    assert sign_disagreement(a, b) == pytest.approx(2 / 5)


N = 8
SHAPE = (synthetic.BOX, synthetic.BOX)
PIXEL_SIZE = synthetic.PIXEL_SIZE


def _gpu():
    try:
        return Gpu()
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"needs CuPy and a CUDA device: {e}")


@pytest.fixture(scope="module", params=["fake_device", pytest.param("gpu", marks=pytest.mark.gpu)])
def device(request):
    return FakeDevice() if request.param == "fake_device" else _gpu()


@pytest.fixture(scope="module")
def host():
    """Images, their CpuSerial spectrum, and CTF columns, all on the host."""
    images, params = synthetic.stack(N), synthetic.params(N)
    return images, pipeline.rfft2(CpuSerial(), images), params.ctf


def _on(backend, images, spectrum, ctf_params):
    return (
        backend.to_device(images),
        backend.to_device(spectrum),
        CtfParams(*(backend.to_device(c) for c in ctf_params)),
    )


def _to_host(backend, x):
    return backend.to_host(x, np.empty(x.shape, x.dtype))


# Each op as (backend, images, spectrum, ctf_params) -> result, keyed by its config tolerance.
OPS = {
    "normalize": lambda b, x, f, p: pipeline.normalize(b, x),
    "rfft2": lambda b, x, f, p: pipeline.rfft2(b, x),
    "irfft2": lambda b, x, f, p: pipeline.irfft2(b, f, SHAPE),
    "lowpass_multiply": lambda b, x, f, p: pipeline.lowpass_multiply(b, f, SHAPE, PIXEL_SIZE),
    "ctf_flip": lambda b, x, f, p: pipeline.ctf_flip(b, f, SHAPE, PIXEL_SIZE, p),
    "pipeline": lambda b, x, f, p: pipeline.process_chunk(b, x, p, PIXEL_SIZE),
}


@pytest.mark.parametrize("op", OPS)
def test_op_agrees_with_cpu_serial(device, host, op, agreement_report):
    expected = OPS[op](CpuSerial(), *host)
    got = _to_host(device, OPS[op](device, *_on(device, *host)))
    assert got.dtype == expected.dtype
    e = errors(got, expected)
    agreement_report(f"{device.name} {op}: {e}")
    assert e.rel_l2 < config.TOLERANCES[op]


def test_stack_agrees_with_cpu_serial(device, agreement_report):
    """Host in, host out through staging and transfers, with a short last chunk."""
    images, params = synthetic.stack(N), synthetic.params(N)
    expected = pipeline.process_stack(images, params, CpuSerial(), chunk_size=3)
    e = errors(pipeline.process_stack(images, params, device, chunk_size=3), expected)
    agreement_report(f"{device.name} process_stack: {e}")
    assert e.rel_l2 < config.TOLERANCES["pipeline"]


def test_reports_ctf_sign_disagreement(device, host, agreement_report):
    """Reported apart from the pipeline error: near CTF zeros float32 rounding may flip a pixel."""
    _, _, ctf_params = host

    def ctf_on(backend, params):
        return ctf(backend.xp, frequency_grid(backend.xp, SHAPE, PIXEL_SIZE), params)

    expected = ctf_on(CpuSerial(), ctf_params)
    got = _to_host(device, ctf_on(device, _on(device, *host)[2]))
    fraction = sign_disagreement(got, expected)
    # Not gated: §6.1 sets any tolerance from the value measured on the GPU, not in advance.
    agreement_report(
        f"{device.name} sign(CTF) disagreement: {fraction:.3e} of {expected.size} pixels"
    )
