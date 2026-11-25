# 11: Kaggle GPU smoke test (M4.5)

**What to build:** A first real-GPU run on Kaggle, done by the user: pinned-SHA install, `CUDA_VISIBLE_DEVICES=0`, `nvidia-smi`, download 2 files, `pytest -m gpu`, the smoke script on ~100 particles. Requires the repo pushed to GitHub at the pinned commit. See spec §6.1, §9.

**Blocked by:** 04 (Download subset script), 09 (Agreement tests and GPU smoke script), 10 (Environment fingerprint and git_sha resolution)

**Status:** ready-for-human

- [ ] `nvidia-smi` shows a T4.
- [ ] Every agreement test passes, or a failing tolerance is reset from measured values under the §6.1 rule and logged in decisions.md.
- [ ] The sign-disagreement fraction is reported.
- [ ] The CUDA-event and perf_counter compute times agree.
- [ ] The resolved `git_sha` matches the pinned `COMMIT`.
