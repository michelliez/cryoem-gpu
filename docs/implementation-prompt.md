# Implementation prompt: CryoEM-GPU

You are building CryoEM-GPU in this repository. The project benchmarks batched cryo-EM particle processing on an NVIDIA T4 against a fair multicore CPU baseline.

Read these before writing any code, in this order of precedence:

1. `docs/decisions.md`: settled decisions, which override everything else.
2. `docs/spec.md`: the technical design, including formulas, architecture, benchmark protocol, tests and figures.
3. `docs/research-empiar-10028-and-colab.md`: dataset facts. The Colab parts are superseded: the project runs on Kaggle.
4. `docs/brief.md`: background and motivation.

Where the spec gives a formula, a name, a path or a number, use it exactly. If something in the spec looks wrong, stop and say so instead of working around it.

## Hard rules

- **Never fabricate results.** Write no speedup, timing or error value anywhere unless it comes from `results/benchmarks.csv` or a saved `results/*.npz`. README numbers are filled in only by `scripts/make_figures.py`. Before real results exist, the README shows placeholders between its markers.
- **You have no GPU.** Build and verify everything on CPU. GPU code paths are exercised locally through the `FakeDevice` backend. The user runs the GPU steps on Kaggle at the hand-off points marked 🛑. Stop at each one and tell the user exactly what to run and what to send back.
- Do not add CuPy as a dependency. Do not write instructions that install CuPy on Kaggle, where `cupy-cuda13x` is preinstalled and a second CuPy package breaks it.
- Do not commit `data/`, downloaded stacks or maps. Commit `results/`.
- No custom CUDA kernels, no frontend, no LLM features, and no claims that this is a reconstruction or a production pipeline.
- All math is written once against `backend.xp`. Only `backend.py` may import `scipy.fft` or `cupy`.
- Tolerances live only in `config.py`.
- The CTF uses the RELION sign: `CTF = +(√(1−A²)·sin χ + A·cos χ)` (spec §3.4).
- Keep code simple. Prefer two fully verified operations over four loosely tested ones.

## Milestones

Finish each milestone, confirm its acceptance criteria by running the commands, and commit before starting the next. In each commit message, state which criteria you verified.

### M0: Scaffold
- Create `pyproject.toml` with the package `cryoem_gpu` under `src/`. Use an MIT license and `requires-python >= 3.10`. Dependencies are `numpy`, `scipy`, `mrcfile` and `matplotlib`. Add extras `[gpu]` (`cupy-cuda13x`) and `[dev]` (`pytest`).
- Add `.gitignore` covering `data/`, caches, `.venv`.
- Add a `LICENSE` file.
- Register the pytest markers `network` and `gpu` in `pyproject.toml`.

**Accept:** `pip install -e .[dev]` succeeds in a fresh venv, and `pytest` runs and collects 0 tests without errors.

### M1: STAR parsing, I/O and download
- `star.py`: implement spec §2.1. `io.py`: implement spec §2.3. `scripts/download_subset.py`: implement spec §2.2.
- Add a test fixture: a short STAR file built from the real header, with a few rows that include a 17-token row.

**Accept:**
- The parser tests pass, including the 17-token row.
- The pixel size test gives 1.34 Å to 1e-3.
- The `network` test downloads `MRC_0601/001`, reads `(97,360,360)` float32, and its STAR row count equals 97. It skips cleanly when offline.
- `download_subset.py --n-files 2` downloads 2 files, writes `subset.star`, and does no re-downloading on a second run.

### M2: Backend shim, normalization, FFT and low-pass
- `backend.py` provides `CpuSerial`, `CpuThreaded` and `FakeDevice`, plus `Gpu` written but untested.
- `filters.py` provides the grid, the n-D raised-cosine low-pass and background normalization.
- `config.py` holds the defaults and the tolerances.

**Accept:**
- Dtype tests pass for float32/complex64 at every stage.
- The low-pass profile tests pass.
- Normalization yields background mean ≈ 0 and std ≈ 1.
- The rfft2→irfft2 round trip is exact to tolerance.
- `CpuThreaded` and `FakeDevice` match `CpuSerial`.

If `CpuThreaded` takes more than about an hour, apply cut #1 (spec §8) and note it in `docs/decisions.md`.

### M3: CTF and independent validation
- `ctf.py`: implement spec §3.4.

**Accept** (spec §6.2):
- λ(300 kV) = 0.019687 Å ± 1e-6.
- For a non-astigmatic case with A = 0, sign changes fall at the analytic zeros of χ.
- CTF(0) = A.
- The U↔V swap with φ + 90° is invariant.
- The phase-flip sign is ±1 only, with sign(0) = +1.

### M4: Fused pipeline and agreement suite
- `pipeline.py`: per-op functions, `process_chunk` (fused) and `process_stack`.
- `tests/test_agreement.py` (marked `gpu`): CPU vs GPU for every op and the pipeline. It reports relative L2 error, RMSE, max abs error, and the fraction of sign(CTF) disagreements.
- `scripts/smoke_gpu.py`: on about 100 real particles, run the agreement suite and print a table. Also print one CUDA-event versus perf_counter cross-check.

**Accept:**
- The fused pipeline equals the sequential per-op composition.
- `process_stack` gives the same result for chunk sizes 1, 7, 500 and N.
- `FakeDevice` runs the full GPU control flow and matches `CpuSerial`.
- Locally, the agreement tests skip cleanly.

### 🛑 M4.5: Kaggle GPU smoke test (the user runs this)
Push to `master`. Give the user a short notebook cell sequence:

1. `COMMIT = "<sha you just pushed>"`, `%env CRYOEM_GPU_COMMIT={COMMIT}`, `%env CUDA_VISIBLE_DEVICES=0`
2. `!nvidia-smi`
3. `!pip install mrcfile git+https://github.com/michelliez/cryoem-gpu@{COMMIT}`
4. Download 2 files.
5. `!python -m pytest -m gpu`
6. `!python scripts/smoke_gpu.py`

Then wait. Fix any GPU-only bug the user reports, and set any agreement tolerance from the measured values using the rule in spec §6.1. Do not continue until the user confirms the smoke test passes.

### M5: Benchmark harness
- `bench.py` and `scripts/run_benchmarks.py`: implement spec §5. That covers the tiers, the experiments `particles_op` and `particles_pipeline` (`volume3d` is added in M8), warm-up, the trial counts, the resumable append-and-flush CSV keyed as in the spec, the fingerprint and `results/environment.json`, the N = 100 dry-run estimate, `--max-minutes`, and running the agreement gate first, refusing to benchmark if it fails.
- The script also saves the `.npz` arrays needed by figures 3–5 (spec §6.3).

**Accept:**
- Using `CpuSerial` and `FakeDevice` on synthetic data:
  - Killing a run partway and rerunning it skips the rows already recorded.
  - Every row has every column.
  - `--max-minutes 0.1` stops between configurations without writing a partial row.
- Any CSV and npz files produced by these tests go to a temporary directory, not to `results/`.

### M6: Kaggle notebook
- Create `notebooks/cryoem_gpu_kaggle.ipynb`. Its cells:
  1. Set `COMMIT`, `CRYOEM_GPU_COMMIT` and `CUDA_VISIBLE_DEVICES=0` (spec §5.4).
  2. Record the environment (`nvidia-smi`, versions, cores, `df -h`, `free -h`).
  3. Install the package from GitHub, pinned to `@{COMMIT}`.
  4. Download the subset and the volume.
  5. Run the agreement gate.
  6. Run `run_benchmarks.py --max-minutes ...`.
  7. Zip `results/` to `/kaggle/working/results.zip`. Do not run `make_figures.py` here: figures are generated locally from the committed CSV (spec §5.5).
- Keep the logic in the package. Notebook cells only call it.

**Accept:** the notebook is valid JSON, runs top to bottom locally with a `--backend fake` / small-N switch, and contains no pasted library code.

### M7: Figures and README numbers
- `scripts/make_figures.py`: implement spec §6.3. It reads only `results/` and writes `results/figures/*.png` and `results/summary.md`. It rewrites the README between the `headline` and `results` markers. The headline sentence uses the full pipeline at N = 10,000, GPU e2e against the faster CPU backend.

**Accept:** on a synthetic CSV, every figure renders, and the README marker blocks are replaced idempotently. Running it twice gives the same file.

### M8: 3D demo
- Run the n-D low-pass on EMD-2660 (spec §5.2 `volume3d`, figure 5), using `mrcfile`, which reads `.gz` directly.

**Accept:** locally, a synthetic 64³ volume matches between `CpuSerial` and `FakeDevice`, and figure 5 renders from a saved npz.

### M9: README
- Write the README following spec §7, in the style of an NVIDIA developer tutorial: concrete, with no hype. Include the marker blocks with placeholders and a Kaggle launch badge.

**Accept:**
- Every number in the README sits inside a generated marker block.
- The limitations section names the measured core count through a marker.
- The README links to `docs/`.

### 🛑 M10: Canonical run (the user runs this)
The user follows spec §5.5: run the notebook with a pinned `COMMIT`, download `results.zip`, check it, and commit `results/` locally. You then run `make_figures.py`, check the README against the CSV, and commit the generated outputs separately. If any figure or claim does not match the data, fix the generator, not the text.

## Cut order

If time runs short, cut in the order given in spec §8. Record every cut in `docs/decisions.md`.
