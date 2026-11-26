import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from cryoem_gpu.io import check_counts, iter_chunks, preload
from cryoem_gpu.star import read_star

pytestmark = pytest.mark.network

BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/"
STAR = "Particles/shiny_2sets.star"
STACK = "Particles/MRC_0601/001_particles_shiny_nb50_new.mrcs"
# data/ is gitignored, so the cache is never committed.
CACHE = Path(__file__).resolve().parents[1] / "data" / "test-cache"


def fetch(rel: str) -> Path:
    """Download BASE/rel into CACHE once; a partial download never lands at the final path."""
    dest = CACHE / rel
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(BASE + rel, timeout=30) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out)
    except urllib.error.HTTPError:
        # The server answered, so we are online and the URL is wrong: fail, don't skip.
        part.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        part.unlink(missing_ok=True)
        pytest.skip(f"EMPIAR unreachable: {e}")
    part.replace(dest)
    return dest


@pytest.fixture(scope="module")
def mrc_0601_001():
    fetch(STACK)
    particles = [p for p in read_star(fetch(STAR)) if p.file == STACK]
    return particles, CACHE


def test_mrc_0601_001_has_97_star_rows_matching_its_header(mrc_0601_001):
    particles, root = mrc_0601_001

    assert len(particles) == 97
    check_counts(particles, root)


def test_mrc_0601_001_preloads_as_97_float32_images(mrc_0601_001):
    particles, root = mrc_0601_001

    stack = preload(particles, root, 97)

    assert stack.shape == (97, 360, 360)
    assert stack.dtype == np.float32
    streamed = np.concatenate(list(iter_chunks(root / STACK, 40)))
    np.testing.assert_array_equal(streamed, stack)
