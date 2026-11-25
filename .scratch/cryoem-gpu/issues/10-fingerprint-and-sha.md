# 10: Environment fingerprint and git_sha resolution (M5a, pulled forward)

**What to build:** The environment fingerprint (`cpu_model, n_cores_usable, gpu_name, driver, cuda_runtime, python, numpy, scipy, cupy, mrcfile`) written once to `environment.json`, and `git_sha` resolution in order: pip `direct_url.json` `vcs_info.commit_id` → `CRYOEM_GPU_COMMIT` env var → `git rev-parse HEAD` → `"unknown"` with a warning, recording which source was used. See spec §5.4.

Pulled ahead of the Kaggle smoke test because M4.5 requires "the resolved `git_sha` matches the pinned `COMMIT`".

**Blocked by:** 01 (Scaffold)

**Status:** ready-for-agent

- [ ] Each SHA source in §5.4 is unit-tested.
- [ ] *(derived, §5.4)* `environment.json` records the fingerprint and which SHA source was used.
- [ ] *(derived, §5.4)* `n_cores_usable` comes from `usable_cores()`.
