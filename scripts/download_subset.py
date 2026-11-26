"""Fetch the EMPIAR-10028 STAR file and the first N MRC_0601 stacks over HTTPS (spec §2.2).

Layout under --out mirrors the server, so subset.star's image paths resolve against --out:

    OUT/Particles/shiny_2sets.star
    OUT/Particles/MRC_0601/001_particles_shiny_nb50_new.mrcs ...
    OUT/subset.star
    OUT/emd_2660.map            (--volume)
"""

import argparse
import gzip
import shutil
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from cryoem_gpu.star import read_star, write_subset

BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/"
VOLUME_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz"
STAR = "Particles/shiny_2sets.star"
HEADER_BYTES = 1024
IMAGE_BYTES = 360 * 360 * 4
# Disk the volume needs while both the .gz (~175 MB) and the 360³ float32 map exist.
VOLUME_BYTES = 175_000_000 + HEADER_BYTES + 360**3 * 4
MB = 1e6


def stack_rel(i: int) -> str:
    return f"Particles/MRC_0601/{i:03d}_particles_shiny_nb50_new.mrcs"


def expected_size(n_particles: int) -> int:
    return HEADER_BYTES + n_particles * IMAGE_BYTES


@contextmanager
def atomic_write(dest: Path) -> Iterator[BinaryIO]:
    """Write through a .part file, so an interrupted write never lands at dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with open(part, "wb") as out:
            yield out
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def fetch(url: str, dest: Path) -> int:
    with urllib.request.urlopen(url, timeout=60) as resp, atomic_write(dest) as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    return dest.stat().st_size


def fetch_stack(base_url: str, out: Path, rel: str, size: int) -> int:
    got = fetch(base_url + rel, out / rel)
    if got != size:
        raise RuntimeError(f"{rel}: downloaded {got} bytes, expected {size}")
    return got


def volume_path(out: Path) -> Path:
    return out / "emd_2660.map"


def fetch_volume(url: str, out: Path) -> int:
    # The map is only ever written whole (atomic_write) and its exact size isn't known
    # in advance, so existence is the validity check.
    dest = volume_path(out)
    if dest.exists():
        return 0
    gz = out / "emd_2660.map.gz"
    got = fetch(url, gz)
    with gzip.open(gz, "rb") as src, atomic_write(dest) as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    gz.unlink()
    return got


def is_valid(path: Path, size: int) -> bool:
    return path.exists() and path.stat().st_size == size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-files", type=int, default=114)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--volume", action="store_true", help="also fetch the EMD-2660 map")
    # Overridable so tests can point at a local server.
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    ap.add_argument("--volume-url", default=VOLUME_URL, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    elapsed = 0.0
    star_path = out / STAR
    if not star_path.exists():
        start = time.perf_counter()
        downloaded += fetch(args.base_url + STAR, star_path)
        elapsed += time.perf_counter() - start

    counts = Counter(p.file for p in read_star(star_path))
    files = [stack_rel(i) for i in range(1, args.n_files + 1)]
    missing = [f for f in files if f not in counts]
    if missing:
        sys.exit(f"not in {STAR}: {', '.join(missing)}")
    sizes = {f: expected_size(counts[f]) for f in files}

    todo = [f for f in files if not is_valid(out / f, sizes[f])]
    need = sum(sizes[f] for f in todo)
    if args.volume and not volume_path(out).exists():
        need += VOLUME_BYTES
    free = shutil.disk_usage(out).free
    print(f"{len(files) - len(todo)} of {len(files)} stacks already valid; "
          f"{len(todo)} to fetch ({need / MB:.1f} MB), {free / MB:.1f} MB free in {out}")
    if need > free:
        sys.exit(f"not enough free disk space: need {need / MB:.1f} MB, have {free / MB:.1f} MB")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rel, got in zip(todo, pool.map(
            lambda f: fetch_stack(args.base_url, out, f, sizes[f]), todo
        )):
            downloaded += got
            print(f"  {rel} ({got / MB:.1f} MB)")

    if args.volume:
        downloaded += fetch_volume(args.volume_url, out)
    elapsed += time.perf_counter() - start

    rows = write_subset(star_path, out / "subset.star", set(files))
    print(f"wrote {out / 'subset.star'} ({rows} particles in {len(files)} files)")
    if downloaded:
        print(f"downloaded {downloaded / MB:.1f} MB in {elapsed:.1f} s "
              f"({downloaded / MB / elapsed:.2f} MB/s)")
    else:
        print("nothing to download")
    print(f"{shutil.disk_usage(out).free / MB:.1f} MB free in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
