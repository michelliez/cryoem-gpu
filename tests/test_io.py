import mrcfile
import numpy as np
import pytest

from cryoem_gpu.io import check_counts, iter_chunks, preload
from cryoem_gpu.star import Particle


def write_stack(root, rel, data):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
    return path


def particle(file, index):
    return Particle(
        file=file,
        index=index,
        defocus_u=20000.0,
        defocus_v=19000.0,
        defocus_angle=30.0,
        voltage=300.0,
        cs=2.0,
        amp_contrast=0.1,
        pixel_size=1.34,
    )


def test_star_rows_matching_stack_nz_pass(tmp_path):
    write_stack(tmp_path, "P/001.mrcs", np.zeros((3, 4, 4)))
    check_counts([particle("P/001.mrcs", i) for i in (1, 2, 3)], tmp_path)


def test_star_row_count_differing_from_stack_nz_raises(tmp_path):
    write_stack(tmp_path, "P/001.mrcs", np.zeros((3, 4, 4)))

    with pytest.raises(ValueError, match=r"P/001\.mrcs.*2.*3"):
        check_counts([particle("P/001.mrcs", i) for i in (1, 2)], tmp_path)


def test_chunks_stream_the_whole_stack_in_order_as_float32(tmp_path):
    data = np.arange(7 * 2 * 3).reshape(7, 2, 3)
    path = write_stack(tmp_path, "s.mrcs", data)

    chunks = list(iter_chunks(path, 3))

    assert [c.shape for c in chunks] == [(3, 2, 3), (3, 2, 3), (1, 2, 3)]
    assert all(c.dtype == np.float32 for c in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), data)


def test_preload_takes_first_n_by_file_then_index_into_one_float32_array(tmp_path):
    # Each image is filled with a value naming its (file, index): 1xx for 001, 2xx for 002.
    write_stack(tmp_path, "P/001.mrcs", [np.full((2, 2), 101), np.full((2, 2), 102)])
    write_stack(tmp_path, "P/002.mrcs", [np.full((2, 2), v) for v in (201, 202, 203)])
    shuffled = [
        particle("P/002.mrcs", 3),
        particle("P/001.mrcs", 2),
        particle("P/002.mrcs", 1),
        particle("P/001.mrcs", 1),
        particle("P/002.mrcs", 2),
    ]

    stack = preload(shuffled, tmp_path, 4)

    assert stack.shape == (4, 2, 2)
    assert stack.dtype == np.float32
    assert stack.flags.c_contiguous
    assert stack[:, 0, 0].tolist() == [101, 102, 201, 202]


def test_preload_of_more_particles_than_available_raises(tmp_path):
    write_stack(tmp_path, "P/001.mrcs", np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="3.*2"):
        preload([particle("P/001.mrcs", i) for i in (1, 2)], tmp_path, 3)


def test_preload_raises_on_row_count_mismatch(tmp_path):
    write_stack(tmp_path, "P/001.mrcs", np.zeros((3, 2, 2)))

    with pytest.raises(ValueError, match="nz=3"):
        preload([particle("P/001.mrcs", 1)], tmp_path, 1)
