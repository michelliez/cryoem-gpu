# Research: EMPIAR-10028 particle data and Colab T4 compatibility

Checked on 2026-11-24. I read the EMPIAR listings and fetched sample files directly (one STAR file, one full `.mrcs` stack, and the byte sizes of four more stacks). The Colab findings come from documentation only. I did not run anything on Colab.

## Verdicts

1. **Particle stacks and CTF STAR file: yes.** EMPIAR-10028 provides 105,247 polished ("shiny") particles, 360 x 360 px, float32, at 1.34 Å/px. They are in 1,081 per-micrograph `.mrcs` stacks. One STAR file in the RELION 1.x/2.x style (no optics groups) gives per-particle defocus U/V, defocus angle, voltage, Cs and amplitude contrast. The pixel size is not stored directly: it is derived from magnification and detector pixel size.
2. **A 10,000-particle subset is about 5.2 GB (4.85 GiB) in roughly 95 to 114 files: yes, it can be downloaded on its own.** Each particle is 518,400 bytes. The sizes I computed match the server byte for byte on every file I checked.
3. **CuPy and mrcfile on the current Colab GPU runtime: very likely yes, but this is from the docs only.** The GPU image is CUDA 13.0 with Python 3.13 and NumPy 2.4.6. `cupy-cuda13x==14.0.1` is preinstalled. `mrcfile` is not preinstalled, but it is pure Python and installs with `pip install mrcfile`. Still to check on a live runtime: the T4 driver version, and whether mrcfile works with NumPy 2.4. mrcfile's changelog lists support only up to NumPy 2.2.

---

## 1. EMPIAR-10028 contents

### Entry metadata

- Title: "Cryo-EM structure of the Plasmodium falciparum 80S ribosome bound to the anti-protozoan drug emetine". Cross-references EMD-2660 and PDB 3j79/3j7a. Total size is 1.2 TB. Last updated 2023-03-28. Source: [EMPIAR API entry](https://www.ebi.ac.uk/empiar/api/entry/10028/).
- The image set "Processed shiny particles" is in directory `data/Particles`. It holds 105,247 images, `image_width`/`image_height` 360, voxel type "32 BIT FLOAT", pixel width 1.34 Å, format MRC. EMPIAR's description: "Particle images are in MRCS image stacks - one each corresponding to the micrograph from which they were picked. The text file shiny_2sets.star describes the picking of the particles from the micrographs." Source: [EMPIAR API entry](https://www.ebi.ac.uk/empiar/api/entry/10028/).
- The other image sets are 1,081 multi-frame movies and 1,081 frame-averaged micrographs (4096 x 4096) in `data/Micrographs/Micrographs_part1` and `Micrographs_part2`. Almost all of the 1.2 TB is these movies. Source: [EMPIAR API entry](https://www.ebi.ac.uk/empiar/api/entry/10028/), [FTP listing of data/Micrographs](https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Micrographs/).
- License: "All data in EMPIAR is freely and publicly available to the global community under the CC0 license." Source: [EMPIAR FAQ](https://www.ebi.ac.uk/empiar/faq). The entry page also shows a CC0 badge ([EMPIAR-10028 page](https://www.ebi.ac.uk/empiar/EMPIAR-10028/)). This answers the brief's redistribution question: CC0 allows redistributing a subset. Citing the paper and EMPIAR is still good practice.

### Directory tree and file names

Base URL: `https://ftp.ebi.ac.uk/empiar/world_availability/10028/` ([listing](https://ftp.ebi.ac.uk/empiar/world_availability/10028/))

```
10028/
  10028.xml                                  7.1K
  data/
    Micrographs/{Micrographs_part1,Micrographs_part2}/   (movies, micrographs, ctffind3 logs, autopick STARs)
    Particles/
      shiny_2sets.star                       27M (28,627,547 bytes)
      MRC_0601/001_particles_shiny_nb50_new.mrcs ... 600_particles_shiny_nb50_new.mrcs   (600 files)
      MRC_1901/001_particles_shiny_nb50_new.mrcs ... 481_particles_shiny_nb50_new.mrcs   (481 files)
```

Sources: [data/Particles/](https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/), [MRC_0601/](https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/MRC_0601/), [MRC_1901/](https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/MRC_1901/). The listed stack sizes range from about 25 MB to 75 MB.

### STAR file: `data/Particles/shiny_2sets.star`

Source: [shiny_2sets.star](https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/shiny_2sets.star), downloaded and parsed on 2026-11-24.

- **Format:** a single `data_` block with one `loop_`. It has no `data_optics` block and no `rlnOpticsGroup` column. This is the pre-RELION-3.1 layout (RELION 1.x/2.x/3.0). The paper used RELION 1.3-beta ([Wong et al. 2014, Methods](https://elifesciences.org/articles/03080); full text read via [Europe PMC PMC4086275](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4086275/fullTextXML)).
- **Columns, in order:** `_rlnVoltage`, `_rlnDefocusU`, `_rlnDefocusV`, `_rlnDefocusAngle`, `_rlnSphericalAberration`, `_rlnDetectorPixelSize`, `_rlnCtfFigureOfMerit`, `_rlnMagnification`, `_rlnAmplitudeContrast`, `_rlnImageName`, `_rlnCoordinateX`, `_rlnCoordinateY`, `_rlnNormCorrection`, `_rlnMicrographName`, `_rlnGroupName`, `_rlnGroupNumber`.
- **Values:** voltage is 300 kV, Cs is 2.0 mm and amplitude contrast is 0.1 on every row. Defocus U/V is in Å and varies per micrograph: the mean of U and V ranges from 8,131 to 38,652 Å. This agrees with the paper's 0.8 to 3.8 µm ([paper Table 1 / Methods](https://elifesciences.org/articles/03080)). The defocus angle is in degrees. CTF parameters come from CTFFIND3 ([paper Methods](https://elifesciences.org/articles/03080)).
- **Pixel size:** there is no `_rlnImagePixelSize` column. The pixel size is `DetectorPixelSize (14 µm) / Magnification (104,478) x 10^4 = 1.34 Å`. This matches the paper ("calibrated magnification of 104,478 (yielding a pixel size of 1.34 Å)") and the EMPIAR metadata.
- **Particle count:** 105,247 rows. This matches EMPIAR (105,247) and the paper ("final refinement ... contained ... 105,247 particles"). MRC_0601 has 53,918 particles and MRC_1901 has 51,328. Each stack holds between 49 and 146 particles, median 96.
- **Image references** look like `000001@Particles/MRC_1901/001_particles_shiny_nb50_new.mrcs`. The index is 1-based and the path is relative to `data/`.
- **Parser trap:** line 5476 (particle `000106@Particles/MRC_1901/052_...mrcs`) has a stray 17th token, `... 52 9`. A strict "16 columns per row" parser will fail on that line or drop it. A parser that takes the first 16 columns reads it correctly. I found this myself in the downloaded file.
- **What the STAR file does not contain:** no orientations (`rlnAngleRot/Tilt/Psi`), no origins, no `rlnPhaseShift`, no `rlnCtfBfactor`. It supports CTF phase flipping. It does not support aligning or rotating particles to a reference using deposited poses.

### Particle stack format, from fetched bytes

- I downloaded the whole stack `MRC_0601/001_particles_shiny_nb50_new.mrcs` (50,285,824 bytes). Its header reads `nx=360, ny=360, nz=97`, `mode=2` (float32), `nsymbt=0` (no extended header), `MAP ` signature, little-endian machine stamp `44 41 00 00`. Source: [file](https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/MRC_0601/001_particles_shiny_nb50_new.mrcs).
- **The header cell size is 0** (`cella = 0,0,0`). Any `voxel_size` that mrcfile reports will be 0, so the pixel size must come from the STAR file or EMPIAR (1.34 Å). The header also has `ispg = 0`, which marks the file as an image stack.
- **Preprocessing:** I computed statistics on the first three particles. Each had mean ≈ 0 and standard deviation ≈ 0.99. Pixels outside a radius of 150 px had mean ≈ -0.03 and standard deviation ≈ 0.94. The stacks therefore already look RELION-normalized, and they have been beam-induced-motion corrected ("polished", [paper Methods](https://elifesciences.org/articles/03080)). RELION does not phase-flip at extraction, so I assume these are not CTF-corrected. That is an inference from the RELION workflow, not something EMPIAR states. A per-particle normalization step in the benchmark will mostly be a no-op on this data. That does not affect benchmark correctness, but it is worth knowing.

## 2. Subset size and download plan

- **Bytes per particle:** 360 x 360 x 4 bytes (float32) = **518,400 bytes**. A stack file is `1024 + n x 518,400` bytes.
- **Cross-check against the server:** I compared that formula, using the per-file counts from the STAR file, with the HTTP `Content-Length` of five files. All five matched exactly:

  | File | n (STAR) | Computed bytes | Server bytes |
  |---|---|---|---|
  | MRC_0601/001 | 97 | 50,285,824 | 50,285,824 (downloaded) |
  | MRC_0601/600 | 101 | 52,359,424 | 52,359,424 |
  | MRC_1901/001 | 103 | 53,396,224 | 53,396,224 |
  | MRC_1901/052 | 109 | 56,506,624 | 56,506,624 |
  | MRC_1901/481 | 109 | 56,506,624 | 56,506,624 |

  Server sizes are from HEAD requests against the [MRC_0601](https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/MRC_0601/) and [MRC_1901](https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/MRC_1901/) directories.
- **All particles:** 105,247 x 518,400 bytes ≈ 54.56 GB (50.8 GiB). The rounded directory listings add up to 51,958 MiB, which agrees.
- **A 10,000-particle subset:** 10,000 x 518,400 bytes = 5.18 GB (4.83 GiB) of pixels. Using whole files:
  - MRC_0601 001 to 114: 114 files, 10,049 particles, 5,209,518,336 bytes (4.85 GiB).
  - MRC_1901 001 to 095: 95 files, 10,089 particles, 5,230,234,880 bytes (4.87 GiB).
- **Subsetting works:** particles are split by micrograph, so you can fetch about 100 files plus the 27 MB STAR file, then filter the STAR rows to the files you downloaded. You do not need to fetch the full 1.2 TB or the 54 GB particle set.
- **Download options** (from the [EMPIAR FAQ](https://www.ebi.ac.uk/empiar/faq)):
  - Individual files over HTTPS/FTP. I used `https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/...`.
  - Aspera: host `fasp.ebi.ac.uk`, port 33001, user `fasp-public`, path `/empiar/world_availability/10028/...`.
  - Globus: collection "EMBL-EBI Public Data", directory `/gridftp/empiar/world_availability/`.
  - HTTP tarballs only work for downloads under 1.5 GB.
- **Download speed:** from this machine, one 50 MB file took 58 s (about 0.9 MB/s), so 5.2 GB would take roughly 1.6 hours over a single stream. Colab's network will be different. Measure it on Colab and use parallel downloads if needed.
- **Memory and storage:** 4.85 GiB of float32 fits on disk. A full complex64 FFT of the same data would be about 9.7 GiB, which is why the brief requires streaming in chunks. Colab does not publish a free-tier disk or RAM figure. Its FAQ only says resources are "dynamic usage limits that sometimes fluctuate" with "standard system memory profile" VMs and at most 12 h runtime ([Colab FAQ](https://research.google.com/colaboratory/faq.html)). Check free disk space and RAM on the live runtime. A smaller fallback is 5,000 particles (about 2.6 GB, about 52 files).

## 3. Colab GPU runtime compatibility (documentation-based)

Primary source: Google's [googlecolab/backend-info](https://github.com/googlecolab/backend-info) repository. Its README warns that the files "do not necessarily reflect the current state of the production environment" and that changes may take "one or two day[s]" to reach runtimes.

| Item | Value | Source (file, last commit date) |
|---|---|---|
| OS / Python (GPU image) | Ubuntu 24.04.4 LTS, Python 3.13.16 | [os-info-gpu.txt](https://github.com/googlecolab/backend-info/blob/main/os-info-gpu.txt), committed 2026-10-21 |
| OS / Python (CPU image, newer) | Ubuntu 24.04.5 LTS, Python 3.13.18 | [os-info.txt](https://github.com/googlecolab/backend-info/blob/main/os-info.txt), fetched 2026-11-24 |
| CUDA | `cuda-toolkit==13.0.3.0` (pip). Apt `cuda-cudart-13-0 13.0.96`, `libcufft-13-0 12.0.0.61`. `torch==2.14.1+cu130`. | [pip-freeze.gpu.txt](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.gpu.txt), committed 2026-11-20; [apt-list-gpu.txt](https://github.com/googlecolab/backend-info/blob/main/apt-list-gpu.txt), fetched 2026-11-24 |
| CuPy | **preinstalled: `cupy-cuda13x==14.0.1`** | pip-freeze.gpu.txt, 2026-11-20 |
| NumPy / SciPy | `numpy==2.4.6`, `scipy==1.16.3` | pip-freeze.gpu.txt, 2026-11-20 |
| mrcfile | **not preinstalled** | pip-freeze.gpu.txt, 2026-11-20 |
| Pinned past runtime "2026.10" | Python 3.13.18, numpy 2.1.3, `cupy-cuda13x==14.0.1`, `cuda-toolkit==13.0.2` | [README](https://github.com/googlecolab/backend-info/blob/main/README.md), committed 2026-11-12; [pip-freeze at aaf59a1](https://github.com/googlecolab/backend-info/blob/aaf59a1929a8b637ef5d41272647c8bee3f8fe7c/pip-freeze.gpu.txt) |
| Pinned past runtime "2026.07" | Ubuntu 22.04, Python 3.12.13, numpy 2.0.2, `cupy-cuda12x==14.0.1`, `cuda-toolkit==12.8.1` | README; [pip-freeze at ec53f93](https://github.com/googlecolab/backend-info/blob/ec53f93fe44a9cab118dc5451527b18e36e1a643/pip-freeze.gpu.txt) |

The GPU os-info file (2026-10-21) is older than the GPU pip-freeze (2026-11-20). The live Python patch version may be 3.13.18 rather than 3.13.16. This does not matter for wheel selection, because both are cp313.

### CuPy

- **Matching wheel:** for CUDA 13.x, use `cupy-cuda13x` ([CuPy install docs](https://docs.cupy.dev/en/stable/install.html), fetched 2026-11-24). Colab already ships it, so **do not `pip install` another CuPy package**. The docs say: "Please make sure that only one CuPy package (cupy or cupy-cudaXX where XX is a CUDA version) is installed" ([CuPy install docs](https://docs.cupy.dev/en/stable/install.html)). Installing `cupy-cuda12x` on top would break this.
- **Requirements** ([CuPy install docs](https://docs.cupy.dev/en/stable/install.html)): compute capability 3.0 or higher, CUDA Toolkit v12.0 to v13.2, Python v3.10 to v3.14, and NumPy tested against v2.0 to v2.4. The T4 is sm_75 and Colab's NumPy is 2.4.6, so both are covered.
- **Package metadata:** `cupy-cuda13x` 14.0.1 (uploaded 2026-02-20) requires `numpy<2.6,>=2.0` and `cuda-pathfinder==1.*,>=1.3.3`. Colab has `cuda-pathfinder==1.9.1`. The latest release is 14.2.0 (2026-08-20), which has cp313 manylinux x86_64 wheels and the same NumPy bound. Sources: [PyPI cupy-cuda13x 14.0.1 JSON](https://pypi.org/pypi/cupy-cuda13x/14.0.1/json), [PyPI cupy-cuda13x](https://pypi.org/project/cupy-cuda13x/). The preinstalled 14.0.1 is enough, and pinning it in the notebook keeps runs reproducible.
- **T4 on CUDA 13:** "CUDA 13.0 supports all NVIDIA architectures from Turing through Grace Blackwell". Only Maxwell, Pascal and Volta were dropped. The minimum Linux driver for 13.0 GA is >= 580.65.06. Source: [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html). Colab's published listings do not show the host driver version.

### mrcfile

- PyPI: latest 1.5.4 (2025-01-22). It ships a pure-Python wheel `mrcfile-1.5.4-py2.py3-none-any.whl`, and its only requirement is `numpy>=1.16.0`, with no upper bound. Python classifiers go up to 3.13. Source: [PyPI mrcfile](https://pypi.org/project/mrcfile/).
- Its docs say it has "no dependencies on any compiled libraries except numpy" and offer a "memory-mapped file option for fast random access to very large files" ([mrcfile README](https://mrcfile.readthedocs.io/en/stable/readme.html)).
- Changelog for 1.5.4: "Add support for Python 3.13 and numpy 2.1 & 2.2". Version 1.5.1 added NumPy 2.0 support, and 1.5.2 and 1.5.3 fixed opening large files and memory-mapped files under NumPy 2.0 ([CHANGELOG.txt](https://github.com/ccpem/mrcfile/blob/master/CHANGELOG.txt)).
- **Conflict assessment:** `pip install mrcfile` will not change Colab's NumPy, because the constraint is only `>=1.16`. There is no binary to build. The remaining risk is that NumPy 2.3 and 2.4 are not in mrcfile's tested list. A small smoke test on the live runtime will settle this.

## Open questions / verify at runtime

1. Run `nvidia-smi` and confirm the GPU is a T4 and the driver is >= 580.65.06, which CUDA 13 requires. Free-tier GPU type is "not guaranteed" ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).
2. Run `pip list | grep -i cupy` and confirm exactly one CuPy package (`cupy-cuda13x`). Then `python -c "import cupy; cupy.show_config(); cupy.fft.fft2(cupy.ones((4,360,360),'f4'))"` to confirm cuFFT loads.
3. Record `python --version`, `numpy.__version__` and `cupy.__version__` on the day of the run, because the listings lag production by one to two days. For reproducibility, consider pinning the notebook to a past runtime such as "2026.10" ([runtime version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html), linked from the backend-info README).
4. Run `pip install mrcfile`. Then open the downloaded stack with `mrcfile.mmap(..., permissive=True)` and `mrcfile.validate`. Confirm the shape is `(97, 360, 360)` and the dtype is float32 under NumPy 2.4.6. Expect `voxel_size == 0`, because the header cell is zeroed.
5. Measure EBI download throughput from Colab, check free disk space (`df -h /content`) and check RAM (`free -h`). Colab does not publish these limits.
6. The STAR parser must tolerate the extra token on line 5476.
7. Assumption: the particles are not CTF-corrected, since RELION does not phase-flip at extraction. For a sanity check, phase-flip a few particles and visually compare 2D power spectra or class averages before trusting the CTF step.
