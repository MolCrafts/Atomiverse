"""Shared pytest fixtures for Atomiverse Python binding tests.

Atomiverse drivers operate directly on a ``molrs.Frame`` — there is no
separate Atomiverse Frame type. Fixtures build ``molrs.Frame`` instances.

Fixtures import from the deep canonical namespaces (the single source of
truth per spec python-api-namespaces-am1); the top-level convenience
re-exports are exercised separately in test_md.py / test_namespaces.py.
"""

import numpy as np
import pytest
import molrs

from atomiverse._binding import extension_available

# Compute imports only when the native extension is present. Pure-import
# tests (test_pure_import.py) never use these fixtures; they run in
# isolated subprocesses.
_HAS_EXT = extension_available()
if _HAS_EXT:
    from atomiverse.md import MD, InitVelocity
    from atomiverse.md.integrator import VelocityVerlet
    from atomiverse.neighbor import NeighborList
    from atomiverse.potentials.pair import LJCut
else:
    MD = InitVelocity = VelocityVerlet = NeighborList = LJCut = None  # type: ignore[misc, assignment]


# Pure-Python tests that must run without the C++ extension.
_PURE_TEST_FILES = frozenset(
    {
        "test_pure_import.py",
        "test_install_cli.py",
    }
)


def pytest_collection_modifyitems(config, items):
    """Skip compute-suite tests when the C++ extension is absent."""
    if _HAS_EXT:
        return
    skip = pytest.mark.skip(reason="C++ extension (_atomiverse) not installed")
    for item in items:
        if item.fspath.basename in _PURE_TEST_FILES:
            continue
        item.add_marker(skip)


def _argon_columns(n_atoms: int):
    """Simple-cubic Ar lattice columns for ``n_atoms`` atoms."""
    side = int(round(n_atoms ** (1 / 3)))
    scale = 3.4  # ~sigma_Ar [A]
    xs, ys, zs = [], [], []
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                xs.append(ix * scale)
                ys.append(iy * scale)
                zs.append(iz * scale)
    box_len = side * scale
    return (
        np.array(xs, dtype=np.float64),
        np.array(ys, dtype=np.float64),
        np.array(zs, dtype=np.float64),
        box_len,
    )


@pytest.fixture
def n_atoms() -> int:
    return 8


@pytest.fixture
def argon_frame(n_atoms: int) -> "molrs.Frame":
    """Single-type Ar molrs.Frame on a simple cubic lattice."""
    xs, ys, zs, box_len = _argon_columns(n_atoms)
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", xs)
    atoms.insert("y", ys)
    atoms.insert("z", zs)
    atoms.insert("mass", np.full(n_atoms, 39.948, dtype=np.float64))
    atoms.insert("atomic_number", np.full(n_atoms, 18, dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molrs.Box.cube(box_len)
    return frame


@pytest.fixture
def full_lj_md():
    """Full LJ MD via categorical setters (mirrors examples/07_md_lj.cpp)."""
    if not _HAS_EXT:
        pytest.skip("C++ extension (_atomiverse) not installed")
    cutoff = 8.5
    sim = MD()
    sim.set_integrator(VelocityVerlet(0.001))
    sim.set_observer(InitVelocity(10.0, 42))
    sim.set_neighbor(NeighborList(cutoff))
    sim.set_evaluator(LJCut(0.0104, 3.4, cutoff))
    return sim
