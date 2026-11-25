# 03: Mmap stack reader and network test (M1b)

**What to build:** Read particle stacks with `mrcfile.mmap(path, permissive=True)` and a streaming chunk reader, plus a preload of the first N particles (fixed order: file number, then 1-based index) into one contiguous host float32 array. Check each file's STAR row count against the stack's header `nz` and raise if they differ. See spec §2, §2.1, §2.3, §6.2.

**Blocked by:** 02 (STAR parser and pixel size)

**Status:** ready-for-agent

- [ ] The `network` test reads `MRC_0601/001` as `(97,360,360)` float32 with 97 STAR rows, and skips offline.
- [ ] *(derived, §6.2)* The network test downloads to a gitignored cache on first use and nothing is committed.
- [ ] *(derived, §2.1)* A STAR/`nz` row-count mismatch raises an error.
