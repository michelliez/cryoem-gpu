# 02: STAR parser and pixel size (M1a)

**What to build:** Parse `shiny_2sets.star` into one record per particle (`file`, `index` 1-based, `defocus_u`, `defocus_v`, `defocus_angle`, `voltage`, `cs`, `amp_contrast`, derived `pixel_size`) with a custom minimal parser for one `data_` block and one `loop_`, no `starfile` dependency. Take the first 16 tokens of each row (line 5476 has a stray 17th). Pixel size = `DetectorPixelSize` / `Magnification` × 1e4; never the MRC header cell. See spec §2.1, §2.

**Blocked by:** 01 (Scaffold)

**Status:** ready-for-agent

- [x] STAR fixture tests pass, including the 17-token row.
- [x] Pixel size = 1.34 Å ± 1e-3.
