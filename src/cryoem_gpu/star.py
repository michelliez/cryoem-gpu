"""Minimal parser for a RELION 1.x-3.0 STAR file: one `data_` block, one `loop_`."""

from dataclasses import dataclass
from os import PathLike


@dataclass(frozen=True)
class Particle:
    file: str
    index: int
    defocus_u: float
    defocus_v: float
    defocus_angle: float
    voltage: float
    cs: float
    amp_contrast: float
    pixel_size: float


_REQUIRED = (
    "rlnImageName",
    "rlnDefocusU",
    "rlnDefocusV",
    "rlnDefocusAngle",
    "rlnVoltage",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
    "rlnDetectorPixelSize",
    "rlnMagnification",
)


def pixel_size(detector_pixel_size_um: float, magnification: float) -> float:
    """Pixel size in Å. The MRC header cell is 0 for this data, so this is the only source."""
    return detector_pixel_size_um / magnification * 1e4


def read_star(path: str | PathLike) -> list[Particle]:
    labels: list[str] = []
    particles: list[Particle] = []
    col: dict[str, int] = {}
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            tokens = line.split()
            if _is_blank_or_block(tokens):
                continue
            if tokens[0].startswith("_"):
                labels.append(tokens[0][1:])
                continue
            if not col:
                col = _column_index(labels)
            # Extra trailing tokens are ignored: shiny_2sets.star line 5476 has a stray 17th.
            if len(tokens) < len(labels):
                raise ValueError(
                    f"{path}: line {lineno} has {len(tokens)} tokens, expected {len(labels)}"
                )
            particles.append(_particle(tokens, col))
    return particles


def write_subset(src: str | PathLike, dest: str | PathLike, files: set[str]) -> int:
    """Copy src to dest, keeping the header and only the data rows whose image is in files.

    Rows are copied verbatim. Returns the number of rows kept.
    """
    labels: list[str] = []
    image_col: int | None = None
    kept = 0
    with open(src) as f, open(dest, "w") as out:
        for line in f:
            tokens = line.split()
            if _is_blank_or_block(tokens):
                out.write(line)
                continue
            if tokens[0].startswith("_"):
                labels.append(tokens[0][1:])
                out.write(line)
                continue
            if image_col is None:
                image_col = _column_index(labels)["rlnImageName"]
            if tokens[image_col].split("@", 1)[1] in files:
                out.write(line)
                kept += 1
    return kept


def _is_blank_or_block(tokens: list[str]) -> bool:
    return not tokens or tokens[0].startswith(("#", "data_")) or tokens[0] == "loop_"


def _column_index(labels: list[str]) -> dict[str, int]:
    missing = [name for name in _REQUIRED if name not in labels]
    if missing:
        raise ValueError(f"STAR file is missing columns: {', '.join(missing)}")
    return {name: labels.index(name) for name in _REQUIRED}


def _particle(tokens: list[str], col: dict[str, int]) -> Particle:
    def num(name: str) -> float:
        return float(tokens[col[name]])

    index, file = tokens[col["rlnImageName"]].split("@", 1)
    return Particle(
        file=file,
        index=int(index),
        defocus_u=num("rlnDefocusU"),
        defocus_v=num("rlnDefocusV"),
        defocus_angle=num("rlnDefocusAngle"),
        voltage=num("rlnVoltage"),
        cs=num("rlnSphericalAberration"),
        amp_contrast=num("rlnAmplitudeContrast"),
        pixel_size=pixel_size(num("rlnDetectorPixelSize"), num("rlnMagnification")),
    )
