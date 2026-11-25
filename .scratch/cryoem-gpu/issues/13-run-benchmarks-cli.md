# 13: run_benchmarks CLI (M5c)

**What to build:** The benchmark runner: `particles_op` and `particles_pipeline` experiments at each N, the `io` tier, a dry run at N = 100 with an estimated total runtime, `--max-minutes M` that stops cleanly between configurations, the agreement gate (runs `gpu` tests first on a GPU backend and refuses to benchmark if any fail), refusal on a GPU backend when the SHA is unknown unless `--allow-unknown-sha`, and small `.npz` outputs for figures 3–4. See spec §5.1–5.4, §6.2, §6.3.

**Blocked by:** 09 (Agreement tests and GPU smoke script), 12 (Timing harness and resumable CSV)

**Status:** ready-for-agent

- [ ] `--max-minutes 0.1` stops between configurations without writing a partial row.
- [ ] *(derived, §5.3)* The dry run prints an estimated total runtime before benchmarking.
- [ ] *(derived, §5.4)* A GPU backend with an unknown SHA is refused unless `--allow-unknown-sha` is passed.
- [ ] *(derived, §6.2)* A failing agreement test blocks benchmarking.
- [ ] *(derived, §6.3)* npz files for figures 3–4 are written under the results dir.
