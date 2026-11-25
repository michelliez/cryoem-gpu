# 12: Timing harness and resumable CSV (M5b)

**What to build:** The timing harness (`perf_counter`, device sync before start and stop, one warm-up per configuration, trials CPU 10/10/5/3 for N = 100/1k/5k/10k and GPU 20, recording median/min/max/n_trials) and the resumable results CSV keyed by `(experiment, op, backend, n_particles, chunk_size, tier)`, appending and flushing each row as it finishes and skipping keys already present. Tiers: `compute` (gpu), `e2e` (all backends), `io` (shared). See spec §5.1, §5.3, §5.4.

**Blocked by:** 08 (Pipeline), 10 (Environment fingerprint and git_sha resolution)

**Status:** ready-for-agent

- [ ] On synthetic data with CpuSerial and FakeDevice: a killed and restarted run skips the recorded rows.
- [ ] Every row has every column.
- [ ] Test outputs go to a temp dir, never to `results/`.
