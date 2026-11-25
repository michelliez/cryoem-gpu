# 14: Kaggle notebook (M6)

**What to build:** The Kaggle notebook: first cell sets `COMMIT` and uses it for both `pip install git+https://github.com/michelliez/cryoem-gpu@{COMMIT}` and `%env CRYOEM_GPU_COMMIT={COMMIT}`; sets `CUDA_VISIBLE_DEVICES=0` before CuPy import; records the environment; picks `/kaggle/working` or `/kaggle/temp` by free space; downloads; runs the agreement gate and benchmarks; zips `results/` as `results.zip`. See spec §4, §5.4, §5.5.

**Blocked by:** 04 (Download subset script), 13 (run_benchmarks CLI)

**Status:** ready-for-agent

- [ ] Valid JSON.
- [ ] Runs top to bottom locally with the fake backend at small N.
- [ ] Cells only call the package and contain no pasted library code.
