import pytest

from cryoem_gpu import config
from cryoem_gpu.star import pixel_size, read_star

HEADER = """
data_

loop_
_rlnVoltage #1
_rlnDefocusU #2
_rlnDefocusV #3
_rlnDefocusAngle #4
_rlnSphericalAberration #5
_rlnDetectorPixelSize #6
_rlnCtfFigureOfMerit #7
_rlnMagnification #8
_rlnAmplitudeContrast #9
_rlnImageName #10
_rlnCoordinateX #11
_rlnCoordinateY #12
_rlnNormCorrection #13
_rlnMicrographName #14
_rlnGroupName #15
_rlnGroupNumber #16
"""

ROW_1 = (
    "300.000000 19685.039062 19317.109375 41.919998 2.000000 14.000000 0.130510 "
    "104478.000000 0.100000 000001@Particles/MRC_0601/001_particles_shiny_nb50_new.mrcs "
    "1830.000000 3456.000000 0.815094 Micrographs/MRC_0601/001.mrc group_1 1"
)
ROW_2 = (
    "300.000000 21000.500000 20500.250000 -12.500000 2.000000 14.000000 0.120000 "
    "104478.000000 0.100000 000002@Particles/MRC_0601/001_particles_shiny_nb50_new.mrcs "
    "100.000000 200.000000 0.900000 Micrographs/MRC_0601/001.mrc group_1 1"
)
# Mirrors line 5476 of shiny_2sets.star: a stray 17th token after GroupNumber.
ROW_17_TOKENS = (
    "300.000000 25000.000000 24000.000000 90.000000 2.000000 14.000000 0.110000 "
    "104478.000000 0.100000 000106@Particles/MRC_1901/052_particles_shiny_nb50_new.mrcs "
    "500.000000 600.000000 0.950000 Micrographs/MRC_1901/052.mrc group_52 52 9"
)


def write_star(tmp_path, *rows):
    path = tmp_path / "particles.star"
    path.write_text(HEADER + "\n".join(rows) + "\n")
    return path


def test_reads_one_record_per_row_with_ctf_fields(tmp_path):
    particles = read_star(write_star(tmp_path, ROW_1, ROW_2))

    assert len(particles) == 2
    first = particles[0]
    assert first.file == "Particles/MRC_0601/001_particles_shiny_nb50_new.mrcs"
    assert first.index == 1
    assert first.defocus_u == pytest.approx(19685.039062)
    assert first.defocus_v == pytest.approx(19317.109375)
    assert first.defocus_angle == pytest.approx(41.919998)
    assert first.voltage == pytest.approx(300.0)
    assert first.cs == pytest.approx(2.0)
    assert first.amp_contrast == pytest.approx(0.1)
    assert particles[1].index == 2
    assert particles[1].defocus_angle == pytest.approx(-12.5)


def test_row_with_stray_17th_token_is_read_from_first_16(tmp_path):
    particles = read_star(write_star(tmp_path, ROW_1, ROW_17_TOKENS, ROW_2))

    assert len(particles) == 3
    stray = particles[1]
    assert stray.file == "Particles/MRC_1901/052_particles_shiny_nb50_new.mrcs"
    assert stray.index == 106
    assert stray.defocus_u == pytest.approx(25000.0)
    assert stray.defocus_angle == pytest.approx(90.0)


def test_pixel_size_derived_from_detector_and_magnification(tmp_path):
    (particle,) = read_star(write_star(tmp_path, ROW_1))

    assert particle.pixel_size == pytest.approx(1.34, abs=config.TOLERANCES["pixel_size"])


def test_pixel_size_formula():
    assert pixel_size(14.0, 104478.0) == pytest.approx(1.34, abs=config.TOLERANCES["pixel_size"])
    assert pixel_size(5.0, 50000.0) == pytest.approx(1.0)


def test_short_row_raises_with_line_number(tmp_path):
    short = " ".join(ROW_1.split()[:10])
    path = write_star(tmp_path, ROW_1, short)

    with pytest.raises(ValueError, match="line"):
        read_star(path)


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "bad.star"
    path.write_text("data_\nloop_\n_rlnVoltage #1\n300.0\n")

    with pytest.raises(ValueError, match="rlnDefocusU"):
        read_star(path)
