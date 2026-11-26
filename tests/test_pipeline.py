import numpy as np
import pytest

from cryoem_gpu import config, pipeline
from cryoem_gpu.backend import CpuSerial, CpuThreaded, FakeDevice
from cryoem_gpu.ctf import CtfParams

PIXEL_SIZE = 1.34
BOX = 360


def _rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _stack(n):
    return np.random.default_rng(0).standard_normal((n, BOX, BOX), dtype=np.float32) + 50


def _params(n):
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


def _to_device(backend, params):
    return CtfParams(*(backend.to_device(col) for col in params.ctf))


@pytest.mark.parametrize("backend", [CpuSerial(), FakeDevice()], ids=lambda b: b.name)
def test_fused_chunk_equals_sequential_per_op_composition(backend):
    host, params = _stack(5), _params(5)
    x = backend.to_device(host)
    ctf_params = _to_device(backend, params)

    y = pipeline.normalize(backend, x)
    f = pipeline.rfft2(backend, y)
    f = pipeline.lowpass_multiply(backend, f, host.shape[-2:], PIXEL_SIZE)
    f = pipeline.ctf_flip(backend, f, host.shape[-2:], PIXEL_SIZE, ctf_params)
    sequential = pipeline.irfft2(backend, f, host.shape[-2:])

    fused = pipeline.process_chunk(backend, x, ctf_params, PIXEL_SIZE)
    assert fused.shape == host.shape
    assert fused.dtype == np.float32
    assert _rel_l2(fused, sequential) < config.TOLERANCES["pipeline"]


N_STACK = 503  # > the default chunk of 500, and prime, so 7 and 500 leave a short last chunk


@pytest.fixture(scope="module")
def stack_and_params():
    return _stack(N_STACK), _params(N_STACK)


@pytest.fixture(scope="module")
def reference(stack_and_params):
    """Each particle through process_chunk alone: what any chunking must reproduce."""
    host, params = stack_and_params
    backend = CpuSerial()
    rows = [
        pipeline.process_chunk(
            backend, host[i : i + 1], CtfParams(*(c[i : i + 1] for c in params.ctf)), PIXEL_SIZE
        )
        for i in range(N_STACK)
    ]
    return np.concatenate(rows)


def _per_particle_rel_l2(a, b):
    return np.linalg.norm(a - b, axis=(1, 2)) / np.linalg.norm(b, axis=(1, 2))


@pytest.mark.parametrize("backend_cls", [CpuSerial, FakeDevice, CpuThreaded], ids=lambda c: c.name)
@pytest.mark.parametrize("chunk_size", [1, 7, 500, N_STACK])
def test_process_stack_is_independent_of_chunk_size(
    stack_and_params, reference, backend_cls, chunk_size
):
    host, params = stack_and_params
    before = host.copy()
    out = pipeline.process_stack(host, params, backend_cls(), chunk_size)
    assert isinstance(out, np.ndarray)
    assert out.shape == host.shape
    assert out.dtype == np.float32
    # Per particle, so a single misplaced or dropped particle cannot hide in the stack's norm.
    assert _per_particle_rel_l2(out, reference).max() < config.TOLERANCES["pipeline"]
    np.testing.assert_array_equal(host, before)


class _RecordingFakeDevice(FakeDevice):
    """FakeDevice that logs each transfer's leading dimension and dtype, and each sync, in order."""

    def __init__(self):
        super().__init__()
        self.log = []

    def to_device(self, host):
        self.log.append(("h2d", host.shape[0], host.dtype))
        return super().to_device(host)

    def to_host(self, x, out):
        self.log.append(("d2h", x.shape[0], x.dtype))
        return super().to_host(x, out)

    def sync(self):
        self.log.append(("sync",))
        super().sync()


def _transfers(rows):
    """One chunk's log: images then every CTF column in, the result out, all float32."""
    h2d = [("h2d", rows, np.float32)] * (1 + len(CtfParams._fields))
    return h2d + [("d2h", rows, np.float32)]


@pytest.mark.parametrize("column_dtype", [np.float32, np.float64])
def test_fake_device_moves_every_chunk_both_ways_in_float32_then_syncs(
    stack_and_params, column_dtype
):
    # float64 columns (np.array of STAR floats) must not reach the device: a dtype change
    # would reallocate the shared pinned staging buffer on every chunk.
    host, params = stack_and_params
    params = params._replace(ctf=CtfParams(*(c.astype(column_dtype) for c in params.ctf)))
    backend = _RecordingFakeDevice()
    pipeline.process_stack(host, params, backend, 200)
    assert backend.log == _transfers(200) + _transfers(200) + _transfers(103) + [("sync",)]
