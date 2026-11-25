# 01: Scaffold (M0)

**What to build:** An installable package skeleton: `pyproject.toml` (src layout, MIT, `>=3.10`, deps numpy/scipy/mrcfile/matplotlib, extras `[gpu]` naming `cupy-cuda13x` and `[dev]`), `.gitignore` (`data/`), LICENSE, and pytest markers `network` and `gpu`. No CuPy in core dependencies (spec §4).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `pip install -e .[dev]` succeeds in a fresh venv.
- [ ] `pytest` collects without errors.
