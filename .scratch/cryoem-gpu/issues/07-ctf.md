# 07: CTF (M3)

**What to build:** Wavelength, astigmatic defocus, χ, CTF and phase-flip sign in the RELION convention (`CTF = √(1−A²)·sin χ + A·cos χ`, no B-factor, no phase shift), computed per chunk on the active backend from per-particle parameters broadcast as `(B,1,1)`. See spec §3.4, including the `ky` orientation risk.

**Blocked by:** 06 (Frequency grid, low-pass, normalization)

**Status:** ready-for-agent

- [x] λ(300 kV) = 0.019687 Å ± 1e-6.
- [x] Non-astigmatic A = 0 sign changes fall at the analytic χ zeros.
- [x] CTF(0) = A.
- [x] U↔V swap with φ + 90° is invariant.
- [x] The flip sign is ±1 only, with sign(0) = +1.
