# 08: Pipeline: per-op, fused, process_stack (M4a)

**What to build:** Per-op functions (normalize, rfft2, irfft2, lowpass_multiply, ctf_flip), the fused `process_chunk` (normalize → rfft2 → multiply by L·sign(CTF) → irfft2, one FFT pair per chunk, "backward" norm), and `process_stack(host_stack, params, backend, chunk_size)`. On device backends each chunk goes through a pinned staging buffer, H2D, compute, D2H into pinned memory, then out to the host output; the whole stack is never pinned. See spec §3.5, §4.

**Blocked by:** 07 (CTF)

**Status:** ready-for-agent

- [x] The fused pipeline equals the sequential per-op composition.
- [x] `process_stack` gives the same result for chunk sizes 1, 7, 500 and N.
- [x] FakeDevice matches CpuSerial through the full GPU control flow.
