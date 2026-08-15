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


#: Bonded-fixture geometry — the C++ unit tests' right-angle 4-tuple
#: (tests/cpu/potentials/test_dihedral_charmm.cpp
#: `make_right_angle_dihedral` / test_improper_harmonic.cpp
#: `make_right_angle_improper`): r1=(0,0,0), r2=(1,0,0), r3=(1,1,0),
#: r4=(1,1,1). Successive bond vectors are mutually orthogonal, so the
#: proper dihedral φ and the improper χ are both exactly π/2 and every
#: torsion energy below is a closed form with no geometry solve.
#: Bond 0-1 is exactly 1.0 Å, which is what makes the FENE check analytic.
_BONDED_XYZ: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
)

#: Cubic cell edge [Å] for the bonded fixture — large enough that no
#: minimum-image wrap can touch a 1 Å bond.
_BONDED_BOX: float = 20.0


@pytest.fixture
def bonded_frame() -> "molrs.Frame":
    """Four carbons carrying one bond, one dihedral and one improper.

    Topology travels as molrs blocks: ``bonds`` (atomi/atomj/type_id),
    ``dihedrals`` and ``impropers`` (atomi..atoml/type_id), all
    ``np.uint32`` — the dtypes ``cpu::from_molrs`` requires
    (src/cpu/frame/molrs_exchange.cpp:147-165, :240-276; connectivity
    columns are read as u32, kinds as ``type_id``).

    Deliberately **no** ``atoms.tag`` column: without one the CPU frame
    keeps its default identity tags, and the 0-based row indices in the
    topology blocks resolve as written. Supplying tags would silently
    re-key the tuples.

    All four sites are carbon (``atomic_number`` = 6, mass 12.011 amu);
    the element is inert here — no bonded style keys on species — but a
    single real element keeps the frame physical.

    @return A 4-atom molrs.Frame in a 20 Å cubic cell with φ = χ = π/2.
    """
    n = len(_BONDED_XYZ)
    atoms = molrs.Block()
    for axis, name in enumerate(("x", "y", "z")):
        atoms.insert(
            name, np.array([site[axis] for site in _BONDED_XYZ], dtype=np.float64)
        )
    atoms.insert("mass", np.full(n, 12.011, dtype=np.float64))
    atoms.insert("atomic_number", np.full(n, 6, dtype=np.uint32))

    bonds = molrs.Block()
    bonds.insert("atomi", np.array([0], dtype=np.uint32))
    bonds.insert("atomj", np.array([1], dtype=np.uint32))
    bonds.insert("type_id", np.array([0], dtype=np.uint32))

    def _four_tuple() -> "molrs.Block":
        block = molrs.Block()
        for column, row in (("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)):
            block.insert(column, np.array([row], dtype=np.uint32))
        block.insert("type_id", np.array([0], dtype=np.uint32))
        return block

    frame = molrs.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds
    frame["dihedrals"] = _four_tuple()
    frame["impropers"] = _four_tuple()
    frame.box = molrs.Box.cube(_BONDED_BOX)
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
