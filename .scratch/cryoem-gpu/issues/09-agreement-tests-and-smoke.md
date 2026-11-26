# 09: Agreement tests and GPU smoke script (M4b)

**What to build:** `gpu`-marked CPU-vs-GPU agreement tests for every op and the pipeline (relative L2, RMSE, max abs error, tolerances from config only), reporting the sign(CTF) disagreement fraction separately from pipeline error; and a smoke script that runs ~100 particles on the GPU and prints GPU `compute` time from `perf_counter` vs CUDA events. See spec §5.3, §6.1, §6.2.

**Blocked by:** 08 (Pipeline)

**Status:** ready-for-agent

- [x] Agreement tests skip cleanly without CuPy.
- [x] *(derived, §6.1)* The sign-disagreement fraction is reported separately from the pipeline error.
- [x] *(derived, §5.3)* The smoke script prints the CUDA-event vs `perf_counter` compute-time comparison and writes nothing to the results CSV.
- [x] *(derived, §6.1)* No tolerance is hard-coded in tests.
