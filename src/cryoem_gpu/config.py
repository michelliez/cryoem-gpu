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
    "normalize": 1e-5,
    "rfft2": 1e-5,
    "irfft2": 1e-5,
    "lowpass_multiply": 1e-5,
    "ctf_flip": 1e-4,
    "pipeline": 1e-4,
}
