"""Honest Python binding surface tests (specs: arch-refactor-08-python,
python-api-namespaces-am1).

The atomiverse wheel must expose the honest MD + CPU Minimize + optional PW + AM1 surface,
with each top-level name an identity re-export of its deep canonical
namespace (pinned per row in test_namespaces.py):

* Canonical LAMMPS-style component names (``PairLJCut``, ``BondHarmonic``,
  ``AngleHarmonic``, ``DihedralMultiHarmonic``) with no legacy aliases
  (``LJ``, ``HarmonicBond``, ``HarmonicAngle``, ``MultiHarmonicDihedral``)
  and no deprecation shims.
* The AM1 semiempirical single-point trio (``AM1``, ``AM1Evaluator``,
  ``AM1Result``) plus the ``single_point`` entry point.
* The old ``optimizer`` facade is gone and FAILS LOUD:
  ``atomiverse.optimizer`` no longer imports, dead package attributes raise
  ``AttributeError``, and the ``_cpp`` extension carries none of the
  never-bound SCF-bag / legacy optimizer symbols (``SCFConfig``, ``XCType``,
  ``scf_gradient``, ...) — the SCF surface is the concrete methods, never a
  config bag.
* Gaussian SCF (spec py-scf-single-point) exposes the concrete methods
  ``RHF`` / ``UHF`` / ``RKS`` / ``UKS`` through ``atomiverse.scf``. They are
  compile-optional (``ATV_BUILD_HF``) and therefore reachable only through
  the lazy attribute + the deep module, never through ``__all__``.
* Plane-wave DFT uses the concrete ``PW`` method (members → build → run).
  PW types are present only when the extension was
  built with ``ATV_BUILD_PW=ON``; the deleted functional entry is absent on
  every build.
* Trajectory output (spec py-recorder-xyz): ``XYZ`` is an unconditional
  top-level name beside ``Thermo`` — the wheel forces ``ATV_BUILD_IO=ON``
  through the CMake module DAG, so it is never capability-probed.
* Monte Carlo (spec py-mc): the three presets ``MC`` / ``NptMC`` / ``GCMC``
  and their four movers (``TranslateMove``, ``VolumeMove``,
  ``InsertDeleteMove``, ``MoveSelector``) are unconditional top-level names
  beside the MD surface — ``ATV_BUILD_MC`` defaults ON, so like ``XYZ`` they
  are never capability-probed and belong in the static ``__all__``.
* Vibrational analysis (spec py-vib): ``Hessian`` and ``HessianResult``
  are unconditional top-level names beside the Monte Carlo surface —
  ``ATV_BUILD_VIB`` defaults ON, so like ``XYZ`` and the MC names they are
  never capability-probed and belong in the static ``__all__``.
* Geometry optimization (spec py-minimize-full): beside ``LBFGS`` /
  ``ForceConverged`` the surface carries the two other optimizers (``CG``,
  ``FIRE``), the four variable-cell relaxers (``IsotropicCellRelax``,
  ``AnisotropicCellRelax``, ``TriclinicCellRelax``,
  ``JointIsotropicRelax``) and the two cell-relax criteria
  (``PressureConverged``, ``StressConverged``). All eight are
  unconditional: ``ATV_BUILD_MINIMIZER`` defaults ON, so like the MC and
  vibration names they are never capability-probed and belong in the
  static ``__all__``.
* Electrostatics (spec py-coulomb-pme): ``PairCoulCut`` and ``CoulLong``
  are unconditional top-level names; ``KSpacePME`` is compile-optional
  (``ATV_BUILD_KSPACE``) and therefore reachable only through the lazy
  attribute + ``atomiverse.longrange``, never through ``__all__``.
* A tree grep gate keeps dead references from creeping back anywhere
  under ``python/``. Forbidden tokens are assembled by string
  concatenation so this file never contains one literally.

Smoke parameters reuse the verified conftest values (Ar LJ
epsilon=0.0104 eV, sigma=3.4 A, mass 39.948 amu; PW on the in-tree
analytic ``gth_name = "GTH-PBE-q4"`` at ecut 8 Ry / 1 k-point: "runs,
underconverged") — no new reference values are introduced.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import molpy

import atomiverse as av
from atomiverse._binding import _cpp, extension_available

# Capability probes are short-circuited on ``extension_available()``: the
# missing-extension sentinel raises ImportError (not AttributeError) from
# ``__getattr__``, so a bare ``hasattr(_cpp, ...)`` would blow up at import
# time on a pure-Python wheel instead of letting conftest skip the file.
# Same idiom as test_md.py.
_HAS_PW: bool = extension_available() and hasattr(_cpp, "PW")
_HAS_KSPACE: bool = extension_available() and hasattr(_cpp, "KSpacePME")
_HAS_HF: bool = extension_available() and hasattr(_cpp, "RHF")

# The concrete Gaussian SCF methods (spec py-scf-single-point).
_SCF_METHODS: tuple[str, ...] = ("RHF", "UHF", "RKS", "UKS")

# ── Shared surface definitions ──────────────────────────────────────

_CANONICAL_NEW_NAMES: tuple[str, ...] = (
    "PairLJCut",
    "BondHarmonic",
    "AngleHarmonic",
    "DihedralMultiHarmonic",
)

# Exact honest top-level surface — STATIC on every build. Compile-optional
# object types (``PW`` / ``PWResult``, ``KSpacePME``) are NOT in ``__all__``:
# they are reachable through the lazy ``__getattr__`` + their deep module
# when the extension carries them, and raise a build-flag-naming
# AttributeError when it does not. Keying this set off a capability probe
# was a latent bug — ``__all__`` is a static list, so the equality assert
# below failed on any PW-enabled build.
_CANONICAL_ALL: frozenset[str] = frozenset(
    {
        "MD",
        "VelocityVerlet",
        "NoseHooverNPT",
        "ParrinelloRahmanNPT",
        "BerendsenBarostat",
        "InitVelocity",
        "NoseHoover",
        "Langevin",
        "CSVR",
        "Minimize",
        "LBFGS",
        "CG",
        "FIRE",
        "IsotropicCellRelax",
        "AnisotropicCellRelax",
        "TriclinicCellRelax",
        "JointIsotropicRelax",
        "ForceConverged",
        "PressureConverged",
        "StressConverged",
        "MinimizeResult",
        "PairLJCut",
        "PairCoulCut",
        "CoulLong",
        "NeighborList",
        "BondHarmonic",
        "AngleHarmonic",
        "DihedralMultiHarmonic",
        "Thermo",
        "XYZ",
        "MC",
        "NptMC",
        "GCMC",
        "TranslateMove",
        "VolumeMove",
        "InsertDeleteMove",
        "MoveSelector",
        "AM1",
        "AM1Evaluator",
        "AM1Result",
        "single_point",
        "Hessian",
        "HessianResult",
        # Force-field catalogue growth (spec py-pair-styles): six pair
        # styles and three bonded styles, all unconditional — they are
        # plain `cpu/potentials/` components with no build flag of their
        # own, exactly like PairLJCut / BondHarmonic beside them.
        "PairMorse",
        "PairBuck",
        "PairBorn",
        "PairYukawa",
        "PairWCA",
        "PairDpd",
        "BondFene",
        "DihedralCharmm",
        "ImproperHarmonic",
    }
)

#: The force-field catalogue growth of spec py-pair-styles. C++ class
#: names verbatim (``cpu::PairMorse``, ``cpu::PairBuck``, ``cpu::PairBorn``,
#: ``cpu::PairYukawa``, ``cpu::PairWCA``, ``cpu::PairDpd``,
#: ``cpu::BondFene``, ``cpu::DihedralCharmm``, ``cpu::ImproperHarmonic``) —
#: including the capitalization: ``PairWCA`` and ``PairDpd`` are spelled the
#: way their headers spell them, not normalized to ``PairWca`` / ``PairDPD``.
_PAIR_STYLE_NAMES: tuple[str, ...] = (
    "PairMorse",
    "PairBuck",
    "PairBorn",
    "PairYukawa",
    "PairWCA",
    "PairDpd",
    "BondFene",
    "DihedralCharmm",
    "ImproperHarmonic",
)

#: The Monte Carlo surface (spec py-mc): three ensemble presets + four
#: movers, all unconditional.
_MC_NAMES: tuple[str, ...] = (
    "MC",
    "NptMC",
    "GCMC",
    "TranslateMove",
    "VolumeMove",
    "InsertDeleteMove",
    "MoveSelector",
)

#: The vibrational-analysis surface (spec py-vib): the finite-difference
#: Hessian driver and its result type, both unconditional.
_VIB_NAMES: tuple[str, ...] = ("Hessian", "HessianResult")

#: The geometry-optimization growth of spec py-minimize-full: the two new
#: optimizers, the four variable-cell relaxers and the two cell-relax
#: criteria. C++ class names verbatim (``cpu::minimizer::*`` /
#: ``cpu::PressureConverged``), all unconditional.
_MINIMIZE_NEW_NAMES: tuple[str, ...] = (
    "CG",
    "FIRE",
    "IsotropicCellRelax",
    "AnisotropicCellRelax",
    "TriclinicCellRelax",
    "JointIsotropicRelax",
    "PressureConverged",
    "StressConverged",
)

_DEAD_PACKAGE_ATTRS: tuple[str, ...] = (
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
    """Canonical-name LJ MD via categorical setters (C++-aligned)."""
    cutoff = 8.5
    sim = av.MD()
    sim.set_integrator(av.VelocityVerlet(0.001))
    sim.set_observer(av.InitVelocity(10.0, 42))
    sim.set_neighbor(av.NeighborList(cutoff))
    sim.set_evaluator(av.PairLJCut(0.0104, 3.4, cutoff))
    return sim


@pytest.fixture
def silicon_frame() -> "molpy.Frame":
    """Tiny 2-atom Si molpy.Frame in a cubic cell (type = Z = 14)."""
    a = 5.43  # Si lattice constant [A]
    frame = molpy.Frame()
    atoms = molpy.Block()
    atoms.insert("x", np.array([0.0, a / 4.0], dtype=np.float64))
    atoms.insert("y", np.array([0.0, a / 4.0], dtype=np.float64))
    atoms.insert("z", np.array([0.0, a / 4.0], dtype=np.float64))
    atoms.insert("mass", np.full(2, 28.0855, dtype=np.float64))
    atoms.insert("atomic_number", np.full(2, 14, dtype=np.uint32))
    frame["atoms"] = atoms
    frame.box = molpy.Box.cube(a)
    return frame


# ── Canonical-name surface ──────────────────────────────────────────


class TestCanonicalSurface:
    @pytest.mark.parametrize("name", _CANONICAL_NEW_NAMES)
    def test_canonical_component_present(self, name: str) -> None:
        """Each canonical LAMMPS-style component name is exported."""
        assert hasattr(av, name), f"canonical component atomiverse.{name} is missing"

    def test_all_is_exactly_the_honest_surface(self) -> None:
        """__all__ is the static MD + minimize + electrostatics + recorder
        + MC + AM1 surface.

        Same 52 names on every build: compile-optional types live behind
        the lazy ``__getattr__``, never in ``__all__``. ``XYZ`` is static
        because ``ATV_BUILD_PYTHON`` forces ``ATV_BUILD_IO=ON`` through the
        CMake module DAG (``cmake/AtvProductModules.cmake:68``); the seven
        Monte Carlo names are static because ``ATV_BUILD_MC`` is ON by
        default (``:25``) and the wheel does not override it, the two
        vibration names (spec py-vib) are static for the same reason
        (``ATV_BUILD_VIB`` ON at ``:27``), and the eight new geometry
        optimization names (spec py-minimize-full) because
        ``ATV_BUILD_MINIMIZER`` is ON by default too. The nine
        force-field styles of spec py-pair-styles are static for the
        plainest reason of all — a pair or bonded style is part of
        ``ATV_BUILD_POTENTIALS`` with no flag of its own, just like
        ``PairLJCut``. None of them is capability-probed.
        """
        assert len(_CANONICAL_ALL) == 52
        assert set(av.__all__) == set(_CANONICAL_ALL)


# ── MD LJ smoke on canonical names ──────────────────────────────────


class TestCanonicalMDSmoke:
    def test_positions_change(
        self, argon_frame: "molpy.Frame", canonical_lj_md: "av.MD"
    ) -> None:
        """10 canonical-name LJ steps move the Ar atoms."""
        x_before = np.asarray(argon_frame["atoms"]["x"]).copy()
        canonical_lj_md.run(argon_frame, 10)
        x_after = np.asarray(argon_frame["atoms"]["x"])
        assert not np.allclose(x_before, x_after), (
            "positions should evolve over 10 MD steps"
        )

    def test_energy_finite(
        self, argon_frame: "molpy.Frame", canonical_lj_md: "av.MD"
    ) -> None:
        """Total energy is exact typed meta, with no result block."""
        canonical_lj_md.run(argon_frame, 10)
        assert "scalars" not in argon_frame.keys()
        assert "stress" not in argon_frame.keys()
        energy = argon_frame.meta["energy"]
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_kinetic_positive(
        self, argon_frame: "molpy.Frame", canonical_lj_md: "av.MD"
    ) -> None:
        """InitVelocity(10 K, seed 42) yields kinetic energy > 0."""
        canonical_lj_md.run(argon_frame, 10)
        kinetic = float(argon_frame.meta["kinetic"])
        assert np.isfinite(kinetic)
        assert kinetic > 0.0


# ── Dead surface fails loud ─────────────────────────────────────────


class TestDeadSurfaceFailsLoud:
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
    def test_explicit_solver_gth_smoke(self, silicon_frame: "molpy.Frame") -> None:
        """The analytic in-tree GTH path runs from Python and returns a
        finite energy [Ry] — 'runs, underconverged' contract at ecut 8 Ry
        and one k-point, no new reference values."""
        pw = av.PW()
        pw.gth_name = "GTH-PBE-q4"  # Si, in-tree analytic GTH; no file
        pw.ecut_ry = 8.0
        pw.k_mesh = 1
        pw.max_iter = 5
        pw.build(silicon_frame)
        result = pw.run(silicon_frame)
        assert result.iterations >= 1
        assert isinstance(result.energy, float)
        assert np.isfinite(result.energy)

    @pytest.mark.skipif(not _HAS_PW, reason="extension built without ATV_BUILD_PW")
    def test_no_pseudopotential_raises(self, silicon_frame: "molpy.Frame") -> None:
        """A plane-wave run has no default pseudopotential: leaving both
        `gth_name` and `upf_path` empty fails loud instead of running a
        bare valence-charge potential that no cutoff converges."""
        pw = av.PW()
        pw.ecut_ry = 8.0
        pw.k_mesh = 1
        with pytest.raises(Exception):
            pw.build(silicon_frame)

    @pytest.mark.skipif(_HAS_PW, reason="extension built with ATV_BUILD_PW")
    @pytest.mark.parametrize("name", ("PW", "PWResult"))
    def test_pw_types_absent_without_backend(self, name: str) -> None:
        """A PW-disabled wheel exports no pretend plane-wave object types."""
        assert not hasattr(av, name)
        assert not hasattr(_cpp, name)

    def test_functional_entry_deleted(self) -> None:
        """The old transient solve function is absent from Python and C++."""
        assert not hasattr(av, "pw_scf")
        assert not hasattr(importlib.import_module("atomiverse.pw"), "pw_scf")
        assert not hasattr(_cpp, "pw_scf")


# ── Gaussian SCF gate (spec py-scf-single-point) ────────────────────


class TestScfSurface:
    """The concrete methods are the surface — never a bag.

    Same capability contract as PW / KSpacePME: reachable by both spellings
    on an ``ATV_BUILD_HF=ON`` extension, a flag-naming ``AttributeError``
    otherwise, and out of the static ``__all__`` on every build.
    """

    @pytest.mark.skipif(not _HAS_HF, reason="extension built without ATV_BUILD_HF")
    @pytest.mark.parametrize("name", _SCF_METHODS)
    def test_scf_method_is_identity_reexport(self, name: str) -> None:
        """``av.RHF`` IS ``atomiverse.scf.RHF`` (one object, two spellings)."""
        scf = importlib.import_module("atomiverse.scf")
        assert getattr(av, name) is getattr(scf, name)

    @pytest.mark.skipif(not _HAS_HF, reason="extension built without ATV_BUILD_HF")
    def test_scf_module_exports_the_four_methods(self) -> None:
        """``atomiverse.scf.__all__`` is exactly the four concrete methods."""
        scf = importlib.import_module("atomiverse.scf")
        assert set(scf.__all__) == set(_SCF_METHODS)

    @pytest.mark.skipif(_HAS_HF, reason="extension built with ATV_BUILD_HF")
    @pytest.mark.parametrize("name", _SCF_METHODS)
    def test_scf_method_absent_names_the_build_flag(self, name: str) -> None:
        """Without the backend the name fails loud and says which flag to
        turn on — no pretend mean-field method."""
        with pytest.raises(AttributeError, match="ATV_BUILD_HF"):
            getattr(av, name)
        assert not hasattr(_cpp, name)

    @pytest.mark.parametrize("name", _SCF_METHODS)
    def test_scf_method_is_never_in_all(self, name: str) -> None:
        """Compile-optional types stay out of the static ``__all__``."""
        assert name not in av.__all__

    def test_no_scf_facade_name(self) -> None:
        """There is no ``SCF`` facade / solver bag on either surface — the
        public API is the concrete method (CLAUDE.md: no `SCFSolver`)."""
        assert not hasattr(av, "SCF")
        assert not hasattr(_cpp, "SCF")


# ── Electrostatics surface (spec py-coulomb-pme) ────────────────────


class TestCoulombSurface:
    """``CoulCut`` / ``CoulLong`` are unconditional; only the reciprocal
    solver is compile-optional."""

    def test_pair_coul_cut_is_identity_reexport(self) -> None:
        """``av.PairCoulCut`` IS ``potentials.pair.CoulCut``."""
        from atomiverse.potentials.pair import CoulCut

        assert av.PairCoulCut is CoulCut

    def test_coul_long_is_identity_reexport(self) -> None:
        """``av.CoulLong`` IS ``longrange.CoulLong``."""
        from atomiverse.longrange import CoulLong

        assert av.CoulLong is CoulLong

    def test_longrange_module_exports_track_the_build(self) -> None:
        """``atomiverse.longrange`` imports on every build and always
        exports ``CoulLong``; ``KSpacePME`` appears iff it was compiled."""
        longrange = importlib.import_module("atomiverse.longrange")
        assert "CoulLong" in longrange.__all__
        assert ("KSpacePME" in longrange.__all__) is _HAS_KSPACE


# ── Recorder surface (spec py-recorder-xyz) ─────────────────────────


class TestRecorderSurface:
    """``Thermo`` and ``XYZ`` are both unconditional top-level names.

    Unlike PW / KSpacePME there is no capability probe here: the Python
    wheel forces ``ATV_BUILD_IO=ON`` through the CMake module DAG, so the
    trajectory recorder is present on every wheel and belongs in the
    static ``__all__``.
    """

    def test_xyz_is_identity_reexport(self) -> None:
        """``av.XYZ`` IS ``recorder.XYZ``."""
        from atomiverse.recorder import XYZ

        assert av.XYZ is XYZ

    def test_recorder_module_exports_xyz(self) -> None:
        """``atomiverse.recorder`` publishes both recorders."""
        recorder = importlib.import_module("atomiverse.recorder")
        assert "Thermo" in recorder.__all__
        assert "XYZ" in recorder.__all__


# ── Monte Carlo surface (spec py-mc) ────────────────────────────────


class TestMonteCarloSurface:
    """The three MC presets and their movers are unconditional names.

    Same contract as the recorder surface, not the PW one: ``ATV_BUILD_MC``
    defaults ON and the wheel never turns it off, so there is no capability
    probe and no build-flag ``AttributeError`` path to test.

    Caveat, deliberately not asserted here: a staged CLI install selects its
    modules explicitly (``python -m atomiverse config md`` sets
    ``ATV_BUILD_MC=OFF``), and on such an extension these names surface a
    bare ``AttributeError`` rather than the flag-naming message ``PW`` /
    ``RHF`` / ``KSpacePME`` get.
    """

    @pytest.mark.parametrize("name", _MC_NAMES)
    def test_mc_name_is_identity_reexport(self, name: str) -> None:
        """``av.<Name>`` IS ``atomiverse.montecarlo.<Name>``."""
        montecarlo = importlib.import_module("atomiverse.montecarlo")
        assert getattr(av, name) is getattr(montecarlo, name)

    def test_montecarlo_module_exports_the_seven_names(self) -> None:
        """``atomiverse.montecarlo.__all__`` is exactly the presets +
        movers — no policy classes, no transaction bag."""
        montecarlo = importlib.import_module("atomiverse.montecarlo")
        assert set(montecarlo.__all__) == set(_MC_NAMES)

    @pytest.mark.parametrize("name", _MC_NAMES)
    def test_mc_name_is_in_all(self, name: str) -> None:
        """Unconditional names belong in the static ``__all__``."""
        assert name in av.__all__


# ── Vibration surface (spec py-vib) ─────────────────────────────────


class TestVibrationSurface:
    """``Hessian`` / ``HessianResult`` are unconditional names.

    Same contract as the Monte Carlo surface, not the PW one:
    ``ATV_BUILD_VIB`` defaults ON (``cmake/AtvProductModules.cmake:27``)
    and the wheel never turns it off, so there is no capability probe here.
    A staged CLI install that leaves ``vib`` out is the same caveat the MC
    block documents.
    """

    @pytest.mark.parametrize("name", _VIB_NAMES)
    def test_vib_name_is_identity_reexport(self, name: str) -> None:
        """``av.<Name>`` IS ``atomiverse.vibration.<Name>``."""
        vibration = importlib.import_module("atomiverse.vibration")
        assert getattr(av, name) is getattr(vibration, name)

    def test_vibration_module_exports_the_two_names(self) -> None:
        """``atomiverse.vibration.__all__`` is exactly the driver and its
        result type — no workspace, no component chain, no config bag."""
        vibration = importlib.import_module("atomiverse.vibration")
        assert set(vibration.__all__) == set(_VIB_NAMES)

    @pytest.mark.parametrize("name", _VIB_NAMES)
    def test_vib_name_is_in_all(self, name: str) -> None:
        """Unconditional names belong in the static ``__all__``."""
        assert name in av.__all__


# ── Minimize surface growth (spec py-minimize-full) ─────────────────


class TestMinimizerSurface:
    """The two extra optimizers, four cell relaxers and two criteria.

    Same contract as the Monte Carlo / vibration surfaces, not the PW one:
    ``ATV_BUILD_MINIMIZER`` defaults ON
    (``cmake/AtvProductModules.cmake:26``) and the wheel never turns it
    off, so there is no capability probe here. Class names are the C++
    ones verbatim — ``CG`` and ``FIRE`` are ``cpu::minimizer::CG`` /
    ``FIRE``, and the criteria are ``cpu::PressureConverged`` /
    ``cpu::StressConverged``.
    """

    @pytest.mark.parametrize("name", _MINIMIZE_NEW_NAMES)
    def test_minimize_name_is_identity_reexport(self, name: str) -> None:
        """``av.<Name>`` IS ``atomiverse.minimizer.<Name>``."""
        minimizer = importlib.import_module("atomiverse.minimizer")
        assert getattr(av, name) is getattr(minimizer, name)

    @pytest.mark.parametrize("name", _MINIMIZE_NEW_NAMES)
    def test_minimize_name_is_in_all(self, name: str) -> None:
        """Unconditional names belong in the static ``__all__``."""
        assert name in av.__all__


# ── Force-field catalogue growth (spec py-pair-styles) ──────────────


class TestPairStyleSurface:
    """The six pair styles and three bonded styles are static names.

    No capability probe and no build-flag ``AttributeError`` path: a
    real-space style is part of ``ATV_BUILD_POTENTIALS``
    (``cmake/AtvProductModules.cmake:30``, ON by default) together with
    ``PairLJCut`` / ``BondHarmonic``, which are already unconditional
    here. Their deep spellings are pinned row by row in
    test_namespaces.py; this class only guards the top-level surface
    list.
    """

    @pytest.mark.parametrize("name", _PAIR_STYLE_NAMES)
    def test_pair_style_name_is_in_all(self, name: str) -> None:
        """Unconditional names belong in the static ``__all__``."""
        assert name in av.__all__

    @pytest.mark.parametrize("name", _PAIR_STYLE_NAMES)
    def test_pair_style_is_bound_by_the_extension(self, name: str) -> None:
        """The compiled extension carries the C++-spelled class itself —
        a Python-level alias of an already-bound style would satisfy
        ``__all__`` while binding no new physics."""
        assert hasattr(_cpp, name)


# ── KSpace gate (mirrors the PW capability contract) ────────────────


class TestKSpaceSurface:
    @pytest.mark.skipif(
        not _HAS_KSPACE, reason="extension built without ATV_BUILD_KSPACE"
    )
    def test_kspace_pme_reachable(self) -> None:
        """A KSPACE-enabled wheel resolves the reciprocal solver by both
        spellings, as one object."""
        from atomiverse.longrange import KSpacePME

        assert av.KSpacePME is KSpacePME

    @pytest.mark.skipif(_HAS_KSPACE, reason="extension built with ATV_BUILD_KSPACE")
    def test_kspace_pme_absent_names_the_build_flag(self) -> None:
        """Without the backend the name fails loud and says which flag to
        turn on — no pretend reciprocal solver."""
        with pytest.raises(AttributeError, match="ATV_BUILD_KSPACE"):
            getattr(av, "KSpacePME")
        assert not hasattr(_cpp, "KSpacePME")

    def test_kspace_pme_is_never_in_all(self) -> None:
        """Compile-optional types stay out of the static ``__all__``."""
        assert "KSpacePME" not in av.__all__


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

# ``RHF`` / ``UHF`` / ``RKS`` / ``UKS`` are NOT scanned for: spec
# py-scf-single-point makes them live names, so ``av.RHF`` is a legitimate
# reference throughout python/ (docs included).
_DEAD_AV_ATTRS: tuple[str, ...] = (
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
    # molpy >= 0.13 re-exports molpy.Frame / molpy.Block by identity, so
    # `molpy.Frame` is no longer a dead reference and is not scanned for.
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
