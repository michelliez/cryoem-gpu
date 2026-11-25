# 17: 3D demo and figure 5 (M8)

**What to build:** The `volume3d` experiment: EMD-2660 low-pass with the n-D raised cosine, timed as H2D, compute and D2H against CPU, saving an npz; and figure 5 with central XY/XZ/YZ slices of original, CPU, GPU and |CPU−GPU| plus the timing breakdown. See spec §5.2, §6.3; cuts #2 and #4 in §8 apply here.

**Blocked by:** 13 (run_benchmarks CLI), 15 (Figures 1–4)

**Status:** ready-for-agent

- [ ] A synthetic 64³ volume matches between CpuSerial and FakeDevice.
- [ ] Figure 5 renders from a saved npz.
