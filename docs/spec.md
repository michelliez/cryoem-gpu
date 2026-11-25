# CryoEM-GPU spec

Status: approved design, 2026-11-25. Sources: [brief.md](brief.md), [research-empiar-10028-and-colab.md](research-empiar-10028-and-colab.md), [decisions.md](decisions.md). Where they conflict, decisions.md wins, then this spec, then the brief.

## 1. Goal

Benchmark batched cryo-EM particle processing on one NVIDIA T4 (Kaggle) against a fair multicore CPU baseline, with verified numerical agreement. The headline result is the full-pipeline end-to-end (transfer-inclusive) speedup on 10,000 particles, taken only from measured data.

Non-goals: reconstruction, 2D classification, alignment, custom CUDA kernels, a frontend, LLM features, any promised speedup.

## 2. Data

| Item | Value |
|---|---|
| Particles | EMPIAR-10028, `data/Particles/MRC_0601/001…114_particles_shiny_nb50_new.mrcs` (10,049 particles) |
| Metadata | `data/Particles/shiny_2sets.star` (27 MB), filtered to those 114 files |
| Base URL | `https://ftp.ebi.ac.uk/empiar/world_availability/10028/` |
| Image | 360×360 float32, already RELION-normalized, not CTF-corrected (inferred) |
| Pixel size | 1.34 Å = `DetectorPixelSize` (µm) / `Magnification` × 1e4. Header cell is 0; never use it |
| Volume | EMD-2660, `https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz` (~175 MB gz), 360³ at 1.34 Å |
| License | CC0. Cite Wong et al. 2014, eLife 3:e03080, EMPIAR-10028, EMD-2660 |

Particle ordering is fixed: by file number, then by 1-based index within the file. A count of N means the first N particles in that order. The benchmark counts are 100, 1,000, 5,000 and 10,000.

### 2.1 STAR parsing

- Write a custom minimal parser for one `data_` block and one `loop_`, with no dependency on the `starfile` package.
- Take the first 16 tokens of each data row. Line 5476 has a stray 17th token. Tolerate it, and cover it with a fixture test.
- Output one record per particle: `file`, `index` (1-based), `defocus_u`, `defocus_v` (Å), `defocus_angle` (degrees), `voltage` (kV), `cs` (mm), `amp_contrast`, `pixel_size` (Å, derived).
- Check each file's STAR row count against the header `nz` of its stack. Raise an error if they differ.

### 2.2 Download

`scripts/download_subset.py --out DIR [--n-files 114] [--workers 4]`

- Fetch the STAR file and the 114 stacks over HTTPS.
- A stack is valid when its size equals `1024 + n·518400`, with n taken from the STAR file. Skip files that are already valid. Re-download anything that is not.
- Write `DIR/subset.star` containing only the rows for the downloaded files.
- Report the throughput and the free disk space.
- `--volume` also fetches the EMD-2660 map.
- `data/` is gitignored. On Kaggle, use whichever of `/kaggle/working` or `/kaggle/temp` has room, after checking at runtime.

### 2.3 Reading

- `mrcfile.mmap(path, permissive=True)` with a streaming chunk reader. T_io is measured with this reader.
- For compute benchmarks, preload the requested N particles into one contiguous host float32 array outside the timed region.

## 3. Pipeline math

Arrays are `(B, 360, 360)`, with axes `[particle, y, x]`, in float32/complex64 throughout. A test asserts the dtypes at every stage on both backends.

### 3.1 Frequency grid

Use `rfft2` over the last two axes. Spatial frequency in 1/Å:

- `kx = rfftfreq(360, d=pixel_size)` along the last axis
- `ky = fftfreq(360, d=pixel_size)` along y
- `s² = kx² + ky²`, `θ = atan2(ky, kx)`

Compute the grid once per (shape, pixel size) on the active backend and cache it.

### 3.2 Normalization (RELION-style background)

For each particle, compute the mean and standard deviation of the pixels at radius > 150 px from the center `(180, 180)`, then return `(img − mean) / std`. Make the radius configurable. This step is close to a no-op on this data. It appears only as a row in the per-op table, never in a chart or the headline.

### 3.3 Low-pass filter (raised cosine)

Work in Fourier-pixel radius `r = s · box · pixel_size`. The cutoff is `r_c = box · pixel_size / 10 Å` (≈ 48.2 px), and the edge width is `w = 5` px.

```
L(r) = 1                                  r ≤ r_c
       0.5·(1 + cos(π·(r − r_c)/w))       r_c < r < r_c + w
       0                                  r ≥ r_c + w
```

Make both the cutoff and the width configurable. The same function, generalized to n dimensions, serves the 3D demo.

### 3.4 CTF (RELION convention)

```
λ(V) = 12.2643247 / sqrt(V · (1 + 0.978466e-6 · V))     Å, V in volts   (300 kV → 0.019687 Å)
Δf(θ) = (U + V)/2 + (U − V)/2 · cos(2·(θ − φ))          U, V in Å, φ = defocus angle in radians
χ = π·λ·Δf(θ)·s² − (π/2)·Cs·λ³·s⁴                       Cs in Å (mm × 1e7)
CTF = √(1 − A²)·sin χ + A·cos χ                          A = amplitude contrast
```

- There is no B-factor and no phase shift, because the STAR file has neither.
- The sign follows RELION. RELION's `-sin(K1·Δf'·s² + K2·s⁴ − K3)` with Δf' = −Δf expands to the + form above, and CTF(s=0) = A > 0. The − form belongs to cryoDRGN and is not used here.
- Phase flip: `F ← F · sign(CTF)`, with sign(0) = +1.
- Compute CTF per chunk on the active backend from per-particle parameters broadcast as `(B,1,1)`. That time counts in the benchmarks.

Orientation risk: whether `ky` points "up" depends on the MRC row order, and getting it wrong mirrors the astigmatism angle. The Thon-ring figure (§6.2) checks this on an astigmatic micrograph. If the predicted ellipse does not line up with the rings, negate `ky` in the CTF grid, then record the change in decisions.md.

### 3.5 Fused pipeline

```
normalize → F = rfft2(x) → F ·= L · sign(CTF) → irfft2(F, s=(360,360))
```

The pipeline uses one FFT pair per chunk. FFT normalization is the backend default ("backward") on both paths, and that convention is identical between them.

## 4. Code architecture

```
src/cryoem_gpu/
  backend.py    Backend: name, xp, rfftn/irfftn, to_device, to_host, sync
                CpuSerial (scipy.fft workers=-1), CpuThreaded (pool; scipy.fft workers=1 inside),
                Gpu (cupy.fft, pinned staging), FakeDevice (NumPy, but routes through the transfer/sync path)
  star.py       STAR parser + pixel-size derivation
  io.py         mmap stack reader, chunk iterator, preload
  filters.py    frequency grids, raised-cosine low-pass (n-D), normalization
  ctf.py        wavelength, defocus, chi, ctf, phase-flip sign
  pipeline.py   per-op functions + fused process_chunk + process_stack(host_stack, params, backend, chunk_size)
  bench.py      timing harness, fingerprint, resumable CSV
  config.py     tolerances, defaults (chunk 500, cutoff 10 Å, edge 5 px, bg radius 150)
scripts/  download_subset.py, run_benchmarks.py, make_figures.py, smoke_gpu.py
notebooks/cryoem_gpu_kaggle.ipynb
tests/  results/  docs/
```

- **One implementation.** All math is written against `backend.xp`. Only `backend.py` imports `scipy.fft` or `cupy`.
- **CuPy is optional.** Import it lazily. `pyproject.toml` has no CuPy dependency, but has an optional `[gpu]` extra naming `cupy-cuda13x`. Never install a second CuPy on Kaggle.
- **GPU device.** The notebook sets `CUDA_VISIBLE_DEVICES=0` before CuPy is imported.
- **GPU transfers.** `process_stack` on the GPU copies each chunk into a pinned staging buffer (`cupyx.empty_pinned`), then H2D, compute, D2H into pinned memory, then copies out to the host output. The pageable↔pinned copies are inside T_e2e. Do not pin the whole stack. Stream overlap is a stretch goal.
- **Threaded CPU.** A `ThreadPoolExecutor` sized to the measured usable cores (`len(os.sched_getaffinity(0))`) processes chunks in parallel into a preallocated output, with `scipy.fft` `workers=1` inside. If this takes more than a small amount of effort, drop it and document NumPy's single-threaded elementwise ops instead (cut #1).
- **FakeDevice.** Lets the CPU-only tests exercise the GPU control flow (staging, transfers, sync, chunk boundaries).
- **Packaging.** MIT license, `requires-python >= 3.10`, `pip install -e .`. The notebook installs with `pip install git+https://github.com/michelliez/cryoem-gpu`.

## 5. Benchmark methodology

### 5.1 Backends and tiers

| Tier | Meaning | Backends |
|---|---|---|
| `compute` | Input already on device, synchronized before and after | gpu |
| `e2e` | Host array in, host array out, including staging and H2D/D2H | cpu_serial, cpu_threaded, gpu |
| `io` | Memmapped streaming read of N particles from disk | shared (one row per N) |

For CPU backends, compute equals e2e, so record only `e2e`. The headline speedup is GPU `e2e` against the faster of `cpu_serial` and `cpu_threaded` at N = 10,000 for the full pipeline.

### 5.2 Experiments

- `particles_op`: normalize, rfft2, irfft2, lowpass_multiply, ctf_flip (CTF compute plus multiply). Run each in isolation at each N.
- `particles_pipeline`: the fused pipeline at each N.
- `volume3d`: EMD-2660 low-pass, split into H2D, compute and D2H, against CPU.
- Stretch: `chunk_sweep`, the GPU pipeline at N = 10,000 with chunk sizes 100/250/500/1000/2000.

### 5.3 Timing protocol

- Use `time.perf_counter()` on every backend. On the GPU, call `cupy.cuda.Device().synchronize()` before starting and before stopping the clock.
- Run one warm-up per configuration, which also builds the cuFFT plans.
- Trials: CPU 10/10/5/3 for N = 100/1k/5k/10k, GPU 20 at every N. Record median, min, max and n_trials.
- Cross-check GPU `compute` against CUDA events once, in `smoke_gpu.py`, and print the comparison. It does not go into the CSV.
- `run_benchmarks.py --max-minutes M` stops cleanly between configurations. It first does a dry run at N = 100 and prints an estimated total runtime.

### 5.4 Results CSV (`results/benchmarks.csv`)

Append each row as soon as its measurement finishes, and flush. The key is `(experiment, op, backend, n_particles, chunk_size, tier)`, and keys already present are skipped, which makes reruns resumable. Columns are the key plus `n_trials, median_s, min_s, max_s, timestamp, git_sha`, plus the fingerprint: `cpu_model, n_cores_usable, gpu_name, driver, cuda_runtime, python, numpy, scipy, cupy, mrcfile`. Also write the fingerprint once to `results/environment.json`.

## 6. Correctness

### 6.1 Tolerances

All tolerances live in `config.py`, with none hard-coded in tests. The metrics are relative L2 error `‖a−b‖/‖b‖`, RMSE and max abs error. The starting tolerance for float32 CPU-vs-GPU relative error is 1e-5 to 1e-4, chosen per op in config.

Known risk: near CTF zero crossings, float32 rounding can make sign(CTF) disagree between backends at a handful of pixels, which flips those Fourier coefficients. The agreement suite must report the fraction of sign disagreements separately from the pipeline error. If sign flips dominate the pipeline error in the M4.5 smoke test, set the tolerance from the measured value and document why. Do not quietly loosen it.

### 6.2 Tests

Local tests run on CPU only. They cover:

- STAR parsing, including the 17-token line.
- Pixel size derivation.
- Dtype preservation.
- The low-pass profile (1 in the passband, 0 beyond the edge, monotone in between).
- Normalization, which gives background mean 0 and std 1.
- CTF closed forms: λ(300 kV) ≈ 0.019687 Å; for a non-astigmatic case with A = 0, zeros of χ at the analytic s; CTF(0) = A; U↔V swap with φ + 90° gives the same result.
- Fused pipeline equals the sequential per-op composition.
- `process_stack` gives the same result for any chunk size, including a size that does not divide N.
- FakeDevice equals CpuSerial, and CpuThreaded equals CpuSerial.

Other test groups:

- `@pytest.mark.network`: the real `MRC_0601/001` stack (97 particles). It downloads to a gitignored cache on first use, skips when offline, and is never committed. It checks shape `(97,360,360)`, dtype float32 and STAR row count.
- `@pytest.mark.gpu`: CPU-vs-GPU agreement for every op and the pipeline. These skip when CuPy or a GPU is missing. On Kaggle they gate the run: `run_benchmarks.py` runs them first and refuses to benchmark if any fail.

### 6.3 Figures (`scripts/make_figures.py`, reads only `results/`)

1. **Headline:** log–log full-pipeline time against N. Lines for the best CPU, GPU e2e and GPU compute.
2. Per-op grouped bars at N = 10,000.
3. Particles: 4 particles spanning low to high defocus. Panels show raw, CPU, GPU and |CPU−GPU| (on its own scale), with max abs error and RMSE in the titles.
4. Thon rings: the mean power spectrum of one micrograph's particles, with predicted CTF zeros overlaid.
5. 3D: central XY/XZ/YZ slices of the original, CPU, GPU and |CPU−GPU|, plus the timing breakdown.

Figures 3 to 5 need array outputs. The notebook saves small `.npz` files for them under `results/`, and `make_figures.py` writes `results/summary.md`. It also rewrites the README between `<!-- BEGIN:results -->` / `<!-- END:results -->` and `<!-- BEGIN:headline -->` / `<!-- END:headline -->` markers. README numbers are never typed by hand.

## 7. README outline

1. The headline sentence and headline chart, both generated.
2. The scientific bottleneck: hundreds of thousands of particles, with FFTs, filtering and CTF repeated over them.
3. GPU implementation choices: one `xp` codebase, batched rFFT, a fused Fourier multiply, pinned staging, chunking.
4. Methodology: tiers, fairness (multithreaded CPU, measured core count), trials, synchronization.
5. Results tables and figures, generated.
6. Correctness.
7. The 3D demo and why a single volume gains less.
8. Limitations:
   - Kaggle's few vCPUs (state the measured count). A multicore workstation would narrow the gap, with no number given.
   - The data is already normalized.
   - This is not a reconstruction or a production pipeline.
9. Reproduce: an Open in Kaggle link and the local commands.
10. Credits and citation.

## 8. Cut order

If time runs short, cut in this order:

1. Threaded CPU, falling back to serial plus documentation.
2. The 3D timing breakdown.
3. Scaling counts, reduced to 1k and 10k.
4. The 3D demo.

Never cut CTF, the agreement tests or the particle benchmarks.
