# 16: Summary and README marker rewrite (M7b)

**What to build:** Generate `results/summary.md` from the results CSV and rewrite the README between `<!-- BEGIN:results -->`/`<!-- END:results -->` and `<!-- BEGIN:headline -->`/`<!-- END:headline -->`. The headline is GPU `e2e` vs the faster of `cpu_serial` and `cpu_threaded` at N = 10,000 for the full pipeline, from measured data only. See spec §1, §5.1, §6.3.

**Blocked by:** 12 (Timing harness and resumable CSV)

**Status:** ready-for-agent

- [ ] Replacing the README markers is idempotent: two runs give an identical file.
- [ ] *(derived, §5.1)* The headline speedup uses the faster CPU backend at N = 10,000.
