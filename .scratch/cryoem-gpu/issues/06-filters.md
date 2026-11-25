# 06: Frequency grid, low-pass, normalization (M2b)

**What to build:** The cached frequency grid (`rfftfreq`/`fftfreq` with `d=pixel_size`, s², θ) per (shape, pixel size) on the active backend; the n-D raised-cosine low-pass in Fourier-pixel radius (cutoff `box·pixel_size/10 Å`, width 5 px, both configurable); and RELION-style background normalization (radius > 150 px from (180,180), configurable). All written against `backend.xp`. See spec §3.1–3.3.

**Blocked by:** 05 (Backends, usable cores, config)

**Status:** ready-for-agent

- [ ] Dtypes are float32/complex64 at every stage.
- [ ] Low-pass profile tests pass (1 in the passband, 0 beyond the edge, monotone in between).
- [ ] Normalized background has mean ≈ 0, std ≈ 1.
