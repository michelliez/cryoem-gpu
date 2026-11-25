# 05: Backends, usable cores, config (M2a)

**What to build:** The backend abstraction (name, `xp`, rfftn/irfftn, to_device, to_host, sync) with CpuSerial (`scipy.fft` workers=-1), CpuThreaded (thread pool sized by `usable_cores()`, `scipy.fft` workers=1 inside), FakeDevice (NumPy routed through the transfer/sync path), and an untested Gpu (lazy CuPy import, pinned staging). `usable_cores()` uses `sched_getaffinity` where available, else `os.cpu_count()`. Config holds tolerances and defaults (chunk 500, cutoff 10 Å, edge 5 px, bg radius 150). Only the backend module imports `scipy.fft` or `cupy`. See spec §4, §6.1. If CpuThreaded costs more than a small effort, apply cut #1 (§8).

**Blocked by:** 01 (Scaffold)

**Status:** ready-for-agent

- [ ] The rfft2→irfft2 round trip is within tolerance.
- [ ] CpuThreaded and FakeDevice match CpuSerial.
- [ ] `usable_cores()` returns ≥ 1 on macOS and Linux.
- [ ] *(derived, §4)* Importing the package without CuPy installed succeeds.
