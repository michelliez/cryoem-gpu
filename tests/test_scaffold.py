from importlib.metadata import requires

from packaging.requirements import Requirement


def _requirement_names(extra=None):
    names = set()
    for req in map(Requirement, requires("cryoem-gpu") or []):
        if extra is None and req.marker is None:
            names.add(req.name)
        elif extra is not None and req.marker is not None and req.marker.evaluate({"extra": extra}):
            names.add(req.name)
    return names


def test_core_dependencies_exclude_cupy():
    assert _requirement_names() == {"numpy", "scipy", "mrcfile", "matplotlib"}


def test_gpu_extra_names_cupy_cuda13x():
    assert _requirement_names("gpu") == {"cupy-cuda13x"}


def test_network_and_gpu_markers_registered(pytestconfig):
    registered = {m.split(":")[0].strip() for m in pytestconfig.getini("markers")}
    assert {"network", "gpu"} <= registered
