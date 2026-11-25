# 04: Download subset script (M1c)

**What to build:** `download_subset.py --out DIR [--n-files 114] [--workers 4] [--volume]` fetches the STAR file and stacks over HTTPS, treats a stack as valid when its size equals `1024 + n·518400` (n from STAR), skips valid files, re-downloads invalid ones, writes `DIR/subset.star` with only the downloaded files' rows, and reports throughput and free disk space. `--volume` also fetches EMD-2660. See spec §2.2.

**Blocked by:** 02 (STAR parser and pixel size)

**Status:** ready-for-agent

- [ ] `--n-files 2` downloads, writes `subset.star`, and downloads nothing on a rerun.
- [ ] *(derived, §2.2)* A truncated stack is detected by size and re-downloaded.
- [ ] *(derived, §2.2)* Throughput and free disk space are reported; `--volume` fetches the EMD-2660 map.
