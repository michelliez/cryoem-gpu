"""Array backends. The only module that imports scipy.fft or cupy."""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft


def usable_cores() -> int:
    """Cores this process may run on: the affinity mask where the OS has one, else cpu_count."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class _Backend:
    def map(self, fn, items):
        """Apply fn to each item, returning the results in order."""
        return [fn(item) for item in items]

    def sync(self):
        pass


class CpuSerial(_Backend):
    """NumPy arrays; scipy.fft spreads each batched FFT over all cores."""

    name = "cpu_serial"
    xp = np
    _workers = -1

    def rfftn(self, x, axes):
        return scipy.fft.rfftn(x, axes=axes, workers=self._workers)

    def irfftn(self, x, s, axes):
        return scipy.fft.irfftn(x, s=s, axes=axes, workers=self._workers)

    def to_device(self, host):
        return host

    def to_host(self, x, out):
        out[...] = x
        return out


class CpuThreaded(CpuSerial):
    """Chunks run in parallel on a pool of usable_cores() threads, each FFT single-threaded."""

    name = "cpu_threaded"
    _workers = 1

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=usable_cores())

    def map(self, fn, items):
        return list(self._pool.map(fn, items))


class _Staged(_Backend):
    """Device backend: every transfer passes through a reused host staging buffer."""

    def __init__(self):
        self._staging = {}

    def _stage(self, direction, shape, dtype):
        """A view of the direction's staging buffer, grown only when too small for shape."""
        size = math.prod(shape)
        buf = self._staging.get(direction)
        if buf is None or buf.dtype != dtype or buf.size < size:
            buf = self._staging[direction] = self._empty_staging(size, dtype)
        return buf[:size].reshape(shape)

    def to_device(self, host):
        staging = self._stage("h2d", host.shape, host.dtype)
        staging[...] = host
        return self._h2d(staging)

    def to_host(self, x, out):
        staging = self._stage("d2h", x.shape, x.dtype)
        self._d2h(x, staging)
        out[...] = staging
        return out


class FakeDevice(_Staged):
    """NumPy standing in for a GPU, so CPU tests exercise staging, transfers and sync."""

    name = "fake_device"
    xp = np
    _workers = CpuSerial._workers
    rfftn = CpuSerial.rfftn
    irfftn = CpuSerial.irfftn

    def _empty_staging(self, shape, dtype):
        return np.empty(shape, dtype)

    def _h2d(self, staging):
        return staging.copy()

    def _d2h(self, x, staging):
        staging[...] = x


class Gpu(_Staged):
    """CuPy on the current CUDA device, with pinned host staging buffers."""

    name = "gpu"

    def __init__(self):
        import cupy
        import cupyx

        super().__init__()
        self.xp = cupy
        self._cupyx = cupyx

    def rfftn(self, x, axes):
        return self.xp.fft.rfftn(x, axes=axes)

    def irfftn(self, x, s, axes):
        return self.xp.fft.irfftn(x, s=s, axes=axes)

    def _empty_staging(self, shape, dtype):
        return self._cupyx.empty_pinned(shape, dtype)

    def _h2d(self, staging):
        x = self.xp.empty(staging.shape, staging.dtype)
        x.set(staging)
        return x

    def _d2h(self, x, staging):
        x.get(out=staging)

    def sync(self):
        self.xp.cuda.Device().synchronize()
