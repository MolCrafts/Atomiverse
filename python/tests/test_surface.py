"""Honest Python binding surface tests (specs: arch-refactor-08-python,
python-api-namespaces-am1).

The atomiverse wheel must expose the honest MD + CPU Minimize + optional PW + AM1 surface,
with each top-level name an identity re-export of its deep canonical
namespace (pinned per row in test_namespaces.py):

* Canonical LAMMPS-style component names (``PairLJCut``, ``BondHarmonic``,
  ``AngleHarmonic``, ``DihedralMultiHarmonic``) with no legacy aliases
  (``LJ``, ``HarmonicBond``, ``HarmonicAngle``, ``MultiHarmonicDihedral``)
  and no deprecation shims.
* The AM1 semiempirical single-point trio (``AM1Config``, ``AM1Evaluator``,
  ``AM1Result``) plus the ``single_point`` entry point.
* The dead SCF / old ``optimizer`` facade is gone and FAILS LOUD:
  ``atomiverse.scf`` / ``atomiverse.optimizer`` no longer import, dead
  package attributes raise ``AttributeError``, and the ``_cpp`` extension
  carries none of the never-bound SCF / legacy optimizer symbols.
* Plane-wave DFT uses the explicit ``PWConfig`` → ``PWSolver.build`` →
  ``PWSolver.run`` lifecycle. PW types are present only when the extension was
  built with ``ATV_BUILD_PW=ON``; the deleted functional entry is absent on
  every build.
* A tree grep gate keeps dead references from creeping back anywhere
  under ``python/``. Forbidden tokens are assembled by string
  concatenation so this file never contains one literally.

Smoke parameters reuse the verified conftest values (Ar LJ
epsilon=0.0104 eV, sigma=3.4 A, mass 39.948 amu; toy PW with empty
upf_path: "runs, unphysical") — no new reference values are introduced.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import molrs

import atomiverse as av
from atomiverse._binding import _cpp

_HAS_PW: bool = hasattr(_cpp, "PWSolver")

# ── Shared surface definitions ──────────────────────────────────────

_CANONICAL_NEW_NAMES: tuple[str, ...] = (
    "PairLJCut",
    "BondHarmonic",
    "AngleHarmonic",
    "DihedralMultiHarmonic",
)

# Exact honest top-level surface. The three PW object types exist only in a
# PW-enabled extension.
_CANONICAL_ALL: frozenset[str] = frozenset(
    {
        "MD",
        "VelocityVerlet",
        "InitVelocity",
        "NoseHoover",
        "Minimize",
        "LBFGS",
        "ForceConverged",
        "MinimizeResult",
        "PairLJCut",
        "NeighborList",
        "BondHarmonic",
        "AngleHarmonic",
        "DihedralMultiHarmonic",
        "Thermo",
        "AM1Config",
        "AM1Evaluator",
        "AM1Result",
        "single_point",
    }
    | ({"PWConfig", "PWSolver", "PWResult"} if _HAS_PW else set())
)

_DEAD_PACKAGE_ATTRS: tuple[str, ...] = (
    "SCF",
    "RHF",
    "UHF",
    "UKS",
    "GeometryOptimizer",
    "Frame",
    "Block",
    "LJ",
    "HarmonicBond",
    "HarmonicAngle",
    "MultiHarmonicDihedral",
    "pw_scf",
)

_DEAD_CPP_ATTRS: tuple[str, ...] = (
    "LBFGSStep",
    "MinimizerPhase",
    "SCFConfig",
    "SpinType",
    "MeanFieldType",
    "XCType",
    "scf_gradient",
    "build_integrals_sto3g",
    "pw_scf",
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def canonical_lj_md() -> "av.MD":
    """Canonical-name LJ MD: VelocityVerlet(dt) + InitVelocity +
    NeighborList + PairLJCut on the force phase (conftest params)."""
    cutoff = 8.5
    sim = av.MD(dt=0.001)
    sim.add(av.InitVelocity(10.0, 42))
    sim.on("force").add(av.NeighborList(cutoff), order=-10).add(
        av.PairLJCut(0.0104, 3.4, cutoff)
    )
    return sim


@pytest.fixture
def silicon_frame() -> "molrs.Frame":
    """Tiny 2-atom Si molrs.Frame in a cubic cell (type = Z = 14)."""
    a = 5.43  # Si lattice constant [A]
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0, a / 4.0], dtype=np.float64))
    atoms.insert("y", np.array([0.0, a / 4.0], dtype=np.float64))
    atoms.insert("z", np.array([0.0, a / 4.0], dtype=np.float64))
    atoms.insert("mass", np.full(2, 28.0855, dtype=np.float64))
    atoms.insert("type", np.full(2, 14, dtype=np.int32))
    frame["atoms"] = atoms
    frame.simbox = molrs.Box.cube(a)
    return frame


# ── Canonical-name surface ──────────────────────────────────────────


class TestCanonicalSurface:
    @pytest.mark.parametrize("name", _CANONICAL_NEW_NAMES)
    def test_canonical_component_present(self, name: str) -> None:
        """Each canonical LAMMPS-style component name is exported."""
        assert hasattr(av, name), f"canonical component atomiverse.{name} is missing"

    def test_all_is_exactly_the_honest_surface(self) -> None:
        """__all__ equals the MD + AM1 base plus optional PW object types."""
        assert len(_CANONICAL_ALL) == (21 if _HAS_PW else 18)
        assert set(av.__all__) == set(_CANONICAL_ALL)


# ── MD LJ smoke on canonical names ──────────────────────────────────


class TestCanonicalMDSmoke:
    def test_positions_change(
        self, argon_frame: "molrs.Frame", canonical_lj_md: "av.MD"
    ) -> None:
        """10 canonical-name LJ steps move the Ar atoms."""
        x_before = np.asarray(argon_frame["atoms"]["x"]).copy()
        canonical_lj_md.run(argon_frame, 10)
        x_after = np.asarray(argon_frame["atoms"]["x"])
        assert not np.allclose(x_before, x_after), (
            "positions should evolve over 10 MD steps"
        )

    def test_energy_finite(
        self, argon_frame: "molrs.Frame", canonical_lj_md: "av.MD"
    ) -> None:
        """Total energy is exact typed meta, with no result block."""
        canonical_lj_md.run(argon_frame, 10)
        assert "scalars" not in argon_frame.keys()
        assert "stress" not in argon_frame.keys()
        energy_meta = argon_frame.meta["energy"]
        assert energy_meta.dtype == "f64"
        energy = float(energy_meta.value)
        assert np.isfinite(energy)

    def test_kinetic_positive(
        self, argon_frame: "molrs.Frame", canonical_lj_md: "av.MD"
    ) -> None:
        """InitVelocity(10 K, seed 42) yields kinetic energy > 0."""
        canonical_lj_md.run(argon_frame, 10)
        kinetic = float(argon_frame.meta["kinetic"].value)
        assert np.isfinite(kinetic)
        assert kinetic > 0.0


# ── Dead surface fails loud ─────────────────────────────────────────


class TestDeadSurfaceFailsLoud:
    def test_scf_module_gone(self) -> None:
        """atomiverse.scf must not exist at all (not merely broken)."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("atomiverse.scf")

    def test_optimizer_module_gone(self) -> None:
        """atomiverse.optimizer must not exist at all."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("atomiverse.optimizer")

    @pytest.mark.parametrize("name", _DEAD_PACKAGE_ATTRS)
    def test_dead_package_attribute_raises(self, name: str) -> None:
        """Legacy / never-working package attributes raise AttributeError."""
        with pytest.raises(AttributeError):
            getattr(av, name)

    @pytest.mark.parametrize("name", _DEAD_CPP_ATTRS)
    def test_dead_extension_symbol_absent(self, name: str) -> None:
        """The compiled extension binds none of the dead SCF/legacy optimizer
        symbols the old facade pretended to call."""
        assert not hasattr(_cpp, name)


# ── PW gate ─────────────────────────────────────────────────────────


class TestPWSurface:
    @pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
    def test_explicit_solver_toy_smoke(self, silicon_frame: "molrs.Frame") -> None:
        """Toy local potential (empty upf_path) runs and returns a
        finite energy [Ry] — 'runs, unphysical' contract, no new
        reference values."""
        config = av.PWConfig()
        config.ecut_ry = 8.0
        config.k_mesh = 1
        config.max_iter = 5
        solver = av.PWSolver(config)
        solver.build(silicon_frame)
        result = solver.run(silicon_frame)
        assert result.iterations >= 1
        assert isinstance(result.energy, float)
        assert np.isfinite(result.energy)

    @pytest.mark.skipif(_HAS_PW, reason="extension built with ATV_BUILD_PW")
    @pytest.mark.parametrize("name", ("PWConfig", "PWSolver", "PWResult"))
    def test_pw_types_absent_without_backend(self, name: str) -> None:
        """A PW-disabled wheel exports no pretend plane-wave object types."""
        assert not hasattr(av, name)
        assert not hasattr(_cpp, name)

    def test_functional_entry_deleted(self) -> None:
        """The old transient solve function is absent from Python and C++."""
        assert not hasattr(av, "pw_scf")
        assert not hasattr(importlib.import_module("atomiverse.pw"), "pw_scf")
        assert not hasattr(_cpp, "pw_scf")


# ── In-tree dead-reference grep gate ────────────────────────────────
#
# Forbidden tokens are assembled by concatenation so that this file
# never contains one literally (the acceptance grep excludes this file;
# the in-test scanner must not self-match either way).

_PYTHON_ROOT: Path = Path(__file__).resolve().parent.parent
_THIS_FILE: Path = Path(__file__).resolve()
_SCAN_SUFFIXES: frozenset[str] = frozenset({".py", ".cpp", ".md"})
_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {"build", "_skbuild", "dist", "venv", "node_modules", "__pycache__"}
)

_DEAD_AV_ATTRS: tuple[str, ...] = (
    "SCF",
    "RHF",
    "UHF",
    "UKS",
    "GeometryOptimizer",
    "Frame",
    "Block",
)


def _forbidden_tokens() -> tuple[str, ...]:
    """Dead-reference tokens, built without literal occurrence here."""
    tokens = ["_cpp" + "." + name for name in _DEAD_CPP_ATTRS]
    tokens.append("Cpu" + "Frame")
    tokens.extend("av" + "." + name for name in _DEAD_AV_ATTRS)
    tokens.extend("atomiverse" + "." + name for name in ("Frame", "Block"))
    tokens.extend("molpy" + "." + name for name in ("Frame", "Block"))
    return tuple(tokens)


def _iter_scan_files() -> Iterator[Path]:
    """Yield scannable files under python/ (excluding this test and
    build artifacts / virtualenv / cache directories)."""
    for dirpath, dirnames, filenames in os.walk(_PYTHON_ROOT):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and not d.endswith(".egg-info")
            and d not in _SKIP_DIR_NAMES
        )
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix not in _SCAN_SUFFIXES:
                continue
            if path.resolve() == _THIS_FILE:
                continue
            yield path


def _find_token(token: str) -> list[str]:
    """Return 'relpath:lineno' hits for ``token`` under python/."""
    hits: list[str] = []
    for path in _iter_scan_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if token not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if token in line:
                hits.append(f"{path.relative_to(_PYTHON_ROOT)}:{lineno}")
    return hits


class TestDeadReferenceGrepGate:
    @pytest.mark.parametrize("token", _forbidden_tokens())
    def test_no_dead_reference_in_tree(self, token: str) -> None:
        """No file under python/ references the dead symbol token."""
        hits = _find_token(token)
        assert not hits, (
            f"forbidden dead-reference token {token!r} found in: " + ", ".join(hits)
        )
