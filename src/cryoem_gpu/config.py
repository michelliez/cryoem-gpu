"""Tolerances and processing defaults. Tests read tolerances from here only."""

CHUNK_SIZE = 500
LOWPASS_CUTOFF_A = 10.0
LOWPASS_EDGE_PX = 5.0
BG_RADIUS_PX = 150.0

# Relative L2 error ‖a−b‖/‖b‖ for float32 data. "roundtrip" bounds rfft2→irfft2 on one
# backend; the per-op entries bound agreement between a backend and CpuSerial
# (spec §6.1 starts at 1e-5..1e-4).
TOLERANCES = {
    "roundtrip": 1e-6,
    # Absolute: float32 frequency grid against a float64 reference, and normalized
    # background mean/std against 0/1.
    "freq_grid": 1e-6,
    "background_stats": 1e-5,
    # Absolute: λ(300 kV) against the spec's 0.019687 Å, and float32 CTF against its
    # closed-form identities (CTF(0) = A, U↔V swap with φ + 90°). χ reaches ~500 rad at
    # the corner of a 360² grid at 31000 Å defocus, so float32 rounding of χ alone moves
    # the CTF by ~6e-5 (measured).
    "wavelength": 1e-6,
    "ctf_closed_form": 1e-4,
    # In grid steps: how far a sampled CTF sign change may sit from the analytic χ zero.
    "ctf_zero_steps": 2,
    # Absolute, Å: STAR-derived pixel size against the spec's 1.34 Å.
    "pixel_size": 1e-3,
    "normalize": 1e-5,
    "rfft2": 1e-5,
    "irfft2": 1e-5,
    "lowpass_multiply": 1e-5,
    "ctf_flip": 1e-4,
    "pipeline": 1e-4,
}
