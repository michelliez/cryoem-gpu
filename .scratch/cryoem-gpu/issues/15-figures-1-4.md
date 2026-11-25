# 15: Figures 1–4 (M7a)

**What to build:** The figure generator, reading only `results/`: (1) headline log–log full-pipeline time vs N with best CPU, GPU e2e and GPU compute; (2) per-op grouped bars at N = 10,000; (3) four particles across defocus with raw/CPU/GPU/|CPU−GPU| panels and max abs error and RMSE in titles; (4) Thon rings with predicted CTF zeros overlaid. Normalization never appears in a chart. See spec §6.3, §3.2.

**Blocked by:** 13 (run_benchmarks CLI)

**Status:** ready-for-agent

- [ ] Every figure renders from a synthetic CSV and npz.
