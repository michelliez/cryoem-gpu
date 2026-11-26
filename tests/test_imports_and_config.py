import ast
import subprocess
import sys
from pathlib import Path

import cryoem_gpu
from cryoem_gpu import config

PACKAGE = Path(cryoem_gpu.__file__).parent


def test_package_imports_and_gpu_fails_cleanly_without_cupy():
    code = (
        "import sys; sys.modules['cupy'] = None; sys.modules['cupyx'] = None\n"
        "import pkgutil, importlib, cryoem_gpu\n"
        "for m in pkgutil.iter_modules(cryoem_gpu.__path__):\n"
        "    importlib.import_module('cryoem_gpu.' + m.name)\n"
        "from cryoem_gpu.backend import Gpu\n"
        "try:\n"
        "    Gpu()\n"
        "except ImportError:\n"
        "    print('gpu-unavailable')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gpu-unavailable"


def _imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


def test_only_backend_imports_scipy_fft_or_cupy():
    offenders = {
        path.name
        for path in PACKAGE.glob("*.py")
        for name in _imported_modules(path)
        if name == "scipy.fft" or name.split(".")[0] in {"cupy", "cupyx"}
    }
    assert offenders <= {"backend.py"}


def test_defaults_match_spec():
    assert config.CHUNK_SIZE == 500
    assert config.LOWPASS_CUTOFF_A == 10.0
    assert config.LOWPASS_EDGE_PX == 5.0
    assert config.BG_RADIUS_PX == 150.0
