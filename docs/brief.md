# CryoEM-GPU

I want to revise the proposed CryoEM-GPU project before generating the Codex implementation prompt.

**Important change: Make batched cryo-EM particle processing the primary benchmark, rather than generic filtering of a single 3D density map.**

My goal is to build a distinctive technical marketing engineering portfolio project for NVIDIA's Digital Biology & Scientific Computing internship, grounded in the actual bottlenecks I've encountered in cryo-EM research at Columbia's Frank Lab.

## Revised project concept

**CryoEM-GPU: Accelerating Cryo-EM Particle Processing with NVIDIA GPUs**

Demonstrate how GPU acceleration improves repetitive scientific image-processing operations across thousands of cryo-EM particle images.

The scientific story should be:

> Cryo-EM workflows process hundreds of thousands of particle images, repeatedly applying Fourier transforms, frequency filtering, and image transformations. Can an NVIDIA GPU accelerate these operations without compromising numerical correctness?

### 1. Primary benchmark: Batched particle processing

Build comparable CPU and GPU implementations for approximately 10,000 publicly available cryo-EM particle images.

Operations:
- Per-particle normalization.
- Batched 2D FFTs and inverse FFTs.
- Fourier-domain low-pass filtering.
- CTF phase flipping, if the dataset includes usable CTF metadata (e.g., a RELION STAR file with defocus, voltage, Cs, and amplitude contrast). If it does, prefer CTF phase flipping over rotations in the MVP: it is cheap, scientifically meaningful, and closer to a real cryo-EM preprocessing step.
- Batched particle rotations or interpolation (stretch goal if CTF correction is included; otherwise part of the MVP).

Use NumPy/SciPy for CPU implementations and CuPy/CuPy-compatible GPU operations for the GPU implementations.

Benchmark each operation individually and the complete pipeline.

The GPU implementation must process images in batches to control memory consumption and minimize unnecessary CPU–GPU transfers.

### 2. Public dataset

Investigate EMPIAR-10028, associated with EMD-2660.

The full EMPIAR dataset is approximately 1.2 TB, so do not download it in full.

First verify:
- Whether we can download approximately 10,000 particle images from a small set of public MRCS files.
- Whether a STAR file with per-particle CTF parameters is included and usable.
- Whether the extracted images require additional preprocessing.
- Whether the subset fits within free Google Colab's storage and memory limits.
- Whether redistribution of the subset is permitted.

If EMPIAR-10028 is impractical, identify a smaller published particle dataset that can be downloaded reproducibly.

Do not use unpublished lab datasets.

### 3. Secondary benchmark: 3D density maps

Include one shorter demonstration using EMD-2660:

- CPU vs GPU 3D Gaussian filtering or FFT-based filtering.
- A visual comparison of the original and processed ribosome map.
- An explanation of why a single-volume GPU benchmark may show limited end-to-end acceleration due to transfer overhead.

This should be a supporting demonstration, not the project's headline.

### 4. Fair benchmarking methodology

Run everything on a free Google Colab NVIDIA T4 GPU when available.

Requirements:
- Record CPU model, available CPU cores, GPU model, CUDA version, and software versions.
- **Make the CPU baseline fair:** use `scipy.fft` with `workers=-1` and multithreaded NumPy/BLAS where applicable. Do not compare a single-threaded CPU path against the GPU.
- Clearly acknowledge that Colab's limited CPU resources (~2 vCPUs) may exaggerate GPU speedups relative to a multicore scientific workstation.
- Benchmark several particle counts, such as 100, 1,000, 5,000, and 10,000.
- Measure both per-operation and full-pipeline latency.
- Report steady-state GPU compute time separately from end-to-end time including transfers.
- Synchronize GPU execution when timing.
- Warm up implementations and use repeated measurements.
- **Cap CPU benchmark time:** use fewer repetitions at large particle counts (e.g., 1–3 trials at 10,000) so slow CPU runs don't exhaust the Colab session. Report the number of trials used for each measurement.
- Verify CPU/GPU numerical agreement with relative error, RMSE, and visual comparisons.
- Avoid inconsistent FFT normalization, boundary conditions, interpolation conventions, and CTF definitions.
- Never fabricate results or promise a particular speedup.

### 5. Memory and session robustness

- 10,000 particles at 360×360 in float32 is about 5 GB, and complex FFT outputs are larger. Colab provides roughly 12 GB of system RAM and 16 GB of GPU memory, so **stream particles in chunks** (e.g., memory-mapped MRCS reads with `mrcfile`) rather than loading the full stack at once, on both CPU and GPU paths. Downsampling or cropping is acceptable if documented.
- Free Colab sessions can disconnect and T4 availability is not guaranteed. **Append each benchmark result to the CSV as soon as it completes**, and make the benchmark script resumable, skipping configurations already recorded.

### 6. Deliverables

Produce a public GitHub repository containing:

1. A reproducible Google Colab notebook with a one-click launch link.
2. CPU and GPU particle-processing implementations.
3. A dataset download/subsetting script.
4. Correctness tests.
5. A performance comparison for individual pipeline operations.
6. A particle-count scaling chart.
7. An end-to-end pipeline benchmark showing compute-only and transfer-inclusive performance.
8. Example processed particle images and numerical difference maps.
9. A brief supporting 3D ribosome-map demonstration.
10. A polished README written like an NVIDIA scientific computing developer tutorial.

**The README should open with one headline chart and one headline result** (e.g., "Full pipeline on 10,000 particles: X× faster end-to-end on a T4," with X taken only from measured results). Then explain the scientific bottleneck, GPU implementation choices, experimental methodology, actual results, limitations, and how to reproduce the benchmarks.

### 7. Project constraints

- Must fit a single weekend of approximately 8–12 focused hours with coding-agent assistance.
- No paid GPU required.
- No complete cryoDRGN or ManifoldEM training runs.
- No custom CUDA kernels unless clearly justified.
- No elaborate frontend.
- No LLM-agent extension in the initial MVP.
- No unpublished research data.
- Do not describe the benchmark as a complete cryo-EM reconstruction or a validated production pipeline.
- Prefer two fully verified particle-processing operations over four poorly validated ones.

### 8. What I need first

Do not immediately write code.

First:
1. Verify the best public particle dataset, whether CTF metadata is available, and a practical download strategy.
2. Check current Colab T4 compatibility with the required libraries.
3. Evaluate whether FFT, filtering, CTF phase flipping, and rotations are feasible within the time budget, and choose which belong in the MVP.
4. Identify the smallest credible MVP.
5. Design the code architecture, chunked data loading, and benchmarking methodology.
6. Produce a detailed, implementation-ready Codex prompt with milestones and acceptance criteria.

**Prioritize scientific relevance, credible benchmark results, and clear technical communication over maximum feature count.**