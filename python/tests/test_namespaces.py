"""Namespaced package layout tests (spec: python-api-namespaces-am1).

The atomiverse package mirrors the C++ backend directory layering as deep
canonical namespaces (``atv.potentials.pair.LJCut``,
``atv.md.integrator.VelocityVerlet``, ``atv.neighbor.NeighborList``,
``atv.montecarlo.GCMC``, ...). The deep path is the single source of truth;
the top-level LAMMPS-style composite names (``atv.PairLJCut``, ...) are
identity re-exports of the SAME class object —
``atv.PairLJCut is atv.potentials.pair.LJCut``.

Also pins the user ruling that backend selection is a ``device=``
constructor parameter (PyTorch convention), never a namespace axis:
``device="cpu"`` is the default and only accepted value in v1; any other
value (e.g. ``"cuda"``) raises ``ValueError`` naming the missing CUDA
bindings BEFORE any compute happens.

No new reference values — the LJ smoke reuses the verified conftest
parameters (Ar, epsilon=0.0104 eV, sigma=3.4 A).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import molpy

import atomiverse as av
from atomiverse._binding import MissingDomainError, _cpp, extension_available
from _domains import built, needs

# Short-circuited on ``extension_available()``: the missing-extension
# sentinel raises ImportError from ``__getattr__``, so a bare
# ``hasattr(_cpp, ...)`` would break collection on a pure-Python wheel
# instead of letting conftest skip the file (idiom from test_md.py).
_HAS_PW: bool = extension_available() and hasattr(_cpp, "PW")
_HAS_HF: bool = extension_available() and hasattr(_cpp, "RHF")

# ── Spec mapping table: (top-level name, deep module, deep name) ────
# One row per entry of the compiled public mapping table.

_BASE_MAPPING: tuple[tuple[str, str, str], ...] = (
    ("MD", "atomiverse.md", "MD"),
    ("VelocityVerlet", "atomiverse.md.integrator", "VelocityVerlet"),
    ("NoseHoover", "atomiverse.md.thermostat", "NoseHoover"),
    ("InitVelocity", "atomiverse.md", "InitVelocity"),
    ("Minimize", "atomiverse.minimizer", "Minimize"),
    ("LBFGS", "atomiverse.minimizer", "LBFGS"),
    ("ForceConverged", "atomiverse.minimizer", "ForceConverged"),
    ("MinimizeResult", "atomiverse.minimizer", "MinimizeResult"),
    # Geometry optimization (spec py-minimize-full): unconditional,
    # ATV_BUILD_MINIMIZER defaults ON. `cpu/minimize/` is one backend-root
    # directory, so all eight share the `atomiverse.minimizer` deep module —
    # the criteria included, even though C++ files them under
    # `cpu/driver/criteria.h` beside ForceConverged, which is already
    # re-exported from here.
    ("CG", "atomiverse.minimizer", "CG"),
    ("FIRE", "atomiverse.minimizer", "FIRE"),
    ("IsotropicCellRelax", "atomiverse.minimizer", "IsotropicCellRelax"),
    ("AnisotropicCellRelax", "atomiverse.minimizer", "AnisotropicCellRelax"),
    ("TriclinicCellRelax", "atomiverse.minimizer", "TriclinicCellRelax"),
    ("JointIsotropicRelax", "atomiverse.minimizer", "JointIsotropicRelax"),
    ("PressureConverged", "atomiverse.minimizer", "PressureConverged"),
    ("StressConverged", "atomiverse.minimizer", "StressConverged"),
    ("NeighborList", "atomiverse.neighbor", "NeighborList"),
    ("PairLJCut", "atomiverse.potentials.pair", "LJCut"),
    ("BondHarmonic", "atomiverse.potentials.bond", "Harmonic"),
    ("AngleHarmonic", "atomiverse.potentials.angle", "Harmonic"),
    ("DihedralMultiHarmonic", "atomiverse.potentials.dihedral", "MultiHarmonic"),
    ("Thermo", "atomiverse.recorder", "Thermo"),
    # Monte Carlo (spec py-mc): unconditional, ATV_BUILD_MC defaults ON.
    ("MC", "atomiverse.montecarlo", "MC"),
    ("NptMC", "atomiverse.montecarlo", "NptMC"),
    ("GCMC", "atomiverse.montecarlo", "GCMC"),
    ("TranslateMove", "atomiverse.montecarlo", "TranslateMove"),
    ("VolumeMove", "atomiverse.montecarlo", "VolumeMove"),
    ("InsertDeleteMove", "atomiverse.montecarlo", "InsertDeleteMove"),
    ("MoveSelector", "atomiverse.montecarlo", "MoveSelector"),
    ("AM1", "atomiverse.semiempirical", "AM1"),
    ("AM1Result", "atomiverse.semiempirical", "AM1Result"),
    ("RM1", "atomiverse.semiempirical", "RM1"),
    ("RM1Result", "atomiverse.semiempirical", "RM1Result"),
    ("MNDO", "atomiverse.semiempirical", "MNDO"),
    ("MNDOResult", "atomiverse.semiempirical", "MNDOResult"),
    ("PM3", "atomiverse.semiempirical", "PM3"),
    ("PM3Result", "atomiverse.semiempirical", "PM3Result"),
    ("Evaluate", "atomiverse.driver", "Evaluate"),
    ("single_point", "atomiverse.driver", "single_point"),
    # Vibrational analysis (spec py-vib): unconditional, ATV_BUILD_VIB
    # defaults ON. `cpu/vibration/` is a backend-root directory, so the
    # deep module is `atomiverse.vibration` — not a member of `md/`.
    ("Hessian", "atomiverse.vibration", "Hessian"),
    ("HessianResult", "atomiverse.vibration", "HessianResult"),
    # Force-field catalogue (spec py-pair-styles): unconditional, part of
    # ATV_BUILD_POTENTIALS like PairLJCut / BondHarmonic above. The deep
    # name drops the category prefix the top-level composite name carries
    # (`PairBuck` -> `potentials.pair.Buck`), exactly as `PairLJCut` ->
    # `potentials.pair.LJCut`; `improper` is a NEW deep module, the fourth
    # bonded category beside bond / angle / dihedral.
    ("PairMorse", "atomiverse.potentials.pair", "Morse"),
    ("PairBuck", "atomiverse.potentials.pair", "Buck"),
    ("PairBorn", "atomiverse.potentials.pair", "Born"),
    ("PairYukawa", "atomiverse.potentials.pair", "Yukawa"),
    ("PairWCA", "atomiverse.potentials.pair", "WCA"),
    ("PairDpd", "atomiverse.potentials.pair", "Dpd"),
    ("BondFene", "atomiverse.potentials.bond", "Fene"),
    ("DihedralCharmm", "atomiverse.potentials.dihedral", "Charmm"),
    ("ImproperHarmonic", "atomiverse.potentials.improper", "Harmonic"),
    # Amber/GAFF styles (atomiverse.forcefield): the fused LJ + real-space
    # Ewald pair is the real-space half of a long-range split, so it lives
    # beside CoulLong.
    ("PairLJCutCoulLong", "atomiverse.longrange", "LJCutCoulLong"),
    ("ImproperCvff", "atomiverse.potentials.improper", "Cvff"),
)

_PW_MAPPING: tuple[tuple[str, str, str], ...] = (
    (
        ("PW", "atomiverse.pw", "PW"),
        ("PWResult", "atomiverse.pw", "PWResult"),
    )
    if _HAS_PW
    else ()
)

# Gaussian SCF (spec py-scf-single-point): four concrete methods, present
# only on an ``ATV_BUILD_HF=ON`` extension.
_SCF_MAPPING: tuple[tuple[str, str, str], ...] = (
    (
        ("RHF", "atomiverse.scf", "RHF"),
        ("UHF", "atomiverse.scf", "UHF"),
        ("RKS", "atomiverse.scf", "RKS"),
        ("UKS", "atomiverse.scf", "UKS"),
    )
    if _HAS_HF
    else ()
)

_MAPPING = _BASE_MAPPING + _PW_MAPPING + _SCF_MAPPING

# The domain each deep module needs to import. driver spans hf and am1 and
# imports on any build; it checks the evaluator's domain when called.
_MODULE_FLAGS: dict[str, tuple[str, ...]] = {
    "atomiverse.md": ("ATV_BUILD_MD",),
    "atomiverse.md.integrator": ("ATV_BUILD_MD",),
    "atomiverse.md.thermostat": ("ATV_BUILD_MD",),
    "atomiverse.recorder": ("ATV_BUILD_MD",),
    "atomiverse.minimizer": ("ATV_BUILD_MINIMIZER",),
    "atomiverse.neighbor": ("ATV_BUILD_POTENTIALS",),
    "atomiverse.potentials": ("ATV_BUILD_POTENTIALS",),
    "atomiverse.potentials.pair": ("ATV_BUILD_POTENTIALS",),
    "atomiverse.potentials.bond": ("ATV_BUILD_POTENTIALS",),
    "atomiverse.potentials.angle": ("ATV_BUILD_POTENTIALS",),
    "atomiverse.potentials.dihedral": ("ATV_BUILD_POTENTIALS",),
    "atomiverse.potentials.improper": ("ATV_BUILD_POTENTIALS",),
    "atomiverse.longrange": ("ATV_BUILD_POTENTIALS",),
    "atomiverse.montecarlo": ("ATV_BUILD_MC",),
    "atomiverse.driver": (),
    "atomiverse.semiempirical": ("ATV_BUILD_SEMIEMPIRICAL",),
    "atomiverse.pw": ("ATV_BUILD_PW",),
    "atomiverse.scf": ("ATV_BUILD_HF",),
    "atomiverse.vibration": ("ATV_BUILD_VIB",),
}

_DEEP_MODULES: tuple[str, ...] = (
    "atomiverse.md",
    "atomiverse.md.integrator",
    "atomiverse.md.thermostat",
    "atomiverse.minimizer",
    "atomiverse.neighbor",
    "atomiverse.potentials",
    "atomiverse.potentials.pair",
    "atomiverse.potentials.bond",
    "atomiverse.potentials.angle",
    "atomiverse.potentials.dihedral",
    "atomiverse.potentials.improper",
    "atomiverse.recorder",
    "atomiverse.montecarlo",
    "atomiverse.driver",
    "atomiverse.semiempirical",
    "atomiverse.pw",
    "atomiverse.scf",
    "atomiverse.vibration",
)


# ── Deep modules importable ─────────────────────────────────────────


class TestDeepModulesImportable:
    @pytest.mark.parametrize(
        "module_name",
        [pytest.param(m, marks=needs(*_MODULE_FLAGS[m])) for m in _DEEP_MODULES],
    )
    def test_deep_module_imports(self, module_name: str) -> None:
        """Each deep canonical module imports on its own."""
        importlib.import_module(module_name)


# ── Top-level names are identity re-exports of the deep truth ───────


class TestTopLevelIdentity:
    def test_mapping_table_matches_compiled_surface(self) -> None:
        """The base has 52 unconditional rows (the ``_MAPPING`` table above:
        drivers, Monte Carlo, vibration, minimization, the force-field
        catalogue, the NDDO methods and the two Amber/GAFF styles of
        atomiverse.forcefield); a PW build contributes two types (PW,
        PWResult) and an HF build the four concrete SCF methods."""
        expected = 52 + (2 if _HAS_PW else 0) + (4 if _HAS_HF else 0)
        assert len(_MAPPING) == expected

    @pytest.mark.parametrize(
        "top_name,deep_module,deep_name",
        [
            pytest.param(
                *row,
                id=row[0],
                marks=needs(*_MODULE_FLAGS[row[1]]),
            )
            for row in _MAPPING
        ],
    )
    def test_top_level_is_deep_object(
        self, top_name: str, deep_module: str, deep_name: str
    ) -> None:
        """atv.<TopName> is the very same object as <deep.path.Name>."""
        deep_obj = getattr(importlib.import_module(deep_module), deep_name)
        assert getattr(av, top_name) is deep_obj, (
            f"atomiverse.{top_name} must be an identity re-export of "
            f"{deep_module}.{deep_name}"
        )


# ── device= parameter (constructor axis, v1 cpu-only) ───────────────


class TestDeviceParameter:
    @needs("ATV_BUILD_MD")
    def test_md_default_device_constructible(self) -> None:
        """MD(dt=...) with no device argument defaults to cpu."""
        from atomiverse.md import MD

        assert MD(dt=0.001) is not None

    @needs("ATV_BUILD_MD")
    def test_md_explicit_cpu_constructible(self) -> None:
        """MD(dt=..., device="cpu") is accepted."""
        from atomiverse.md import MD

        assert MD(dt=0.001, device="cpu") is not None

    @needs("ATV_BUILD_MD")
    def test_md_cpu_device_runs_lj_smoke(self, argon_frame: "molpy.Frame") -> None:
        """MD(device="cpu") runs the conftest LJ smoke to a finite energy."""
        from atomiverse.md import MD, InitVelocity
        from atomiverse.neighbor import NeighborList
        from atomiverse.potentials.pair import LJCut

        from atomiverse.md.integrator import VelocityVerlet

        cutoff = 8.5
        sim = MD(device="cpu")
        sim.set_integrator(VelocityVerlet(0.001))
        sim.set_observer(InitVelocity(10.0, 42))
        sim.set_neighbor(NeighborList(cutoff))
        sim.set_evaluator(LJCut(0.0104, 3.4, cutoff))
        sim.run(argon_frame, 5)
        energy = float(argon_frame.meta["energy"])
        assert np.isfinite(energy)

    @needs("ATV_BUILD_MD")
    def test_md_cuda_device_raises_value_error(self) -> None:
        """MD(device="cuda") raises ValueError naming CUDA."""
        from atomiverse.md import MD

        if hasattr(_cpp, "_CudaMD"):
            assert MD(dt=0.001, device="cuda").device == "cuda"
        else:
            with pytest.raises(ValueError, match="(?i)cuda"):
                MD(dt=0.001, device="cuda")

    @needs("ATV_BUILD_MINIMIZER")
    def test_minimize_cuda_device_raises_value_error(self) -> None:
        """Minimize(device="cuda") fails loud while GPU work is deferred."""
        from atomiverse.minimizer import Minimize

        if hasattr(_cpp, "_CudaMinimize"):
            assert Minimize(device="cuda").device == "cuda"
        else:
            with pytest.raises(ValueError, match="(?i)gpu|cuda"):
                Minimize(device="cuda")

    @needs("ATV_BUILD_SEMIEMPIRICAL")
    def test_single_point_cuda_device_raises_before_compute(
        self, argon_frame: "molpy.Frame"
    ) -> None:
        """single_point(..., device="cuda") raises ValueError BEFORE any
        compute: the argon frame (type=0) would itself fail AM1 evaluation
        with RuntimeError, so getting ValueError proves the device check
        precedes the calculation."""
        from atomiverse.driver import single_point
        from atomiverse.semiempirical import AM1

        with pytest.raises(ValueError, match="(?i)cuda"):
            single_point(argon_frame, AM1(), device="cuda")


# ── A module the extension was built without fails naming its flag ──

#: One top-level name per product module.
_DOMAIN_NAMES: dict[str, str] = {
    "PairLJCut": "ATV_BUILD_POTENTIALS",
    "MD": "ATV_BUILD_MD",
    "Minimize": "ATV_BUILD_MINIMIZER",
    "MC": "ATV_BUILD_MC",
    "Hessian": "ATV_BUILD_VIB",
    "AM1": "ATV_BUILD_SEMIEMPIRICAL",
    "RHF": "ATV_BUILD_HF",
    "PW": "ATV_BUILD_PW",
    "KSpacePME": "ATV_BUILD_KSPACE",
}


def _absent(*flags: str) -> pytest.MarkDecorator:
    """Run only on an extension that is present but lacks ``flags``."""
    return pytest.mark.skipif(
        not extension_available() or built(*flags),
        reason="needs an extension built without " + " and ".join(flags),
    )


class TestMissingDomain:
    @pytest.mark.parametrize(
        "module_name",
        [
            pytest.param(m, marks=_absent(*_MODULE_FLAGS[m]))
            for m in _DEEP_MODULES
            if _MODULE_FLAGS[m]
        ],
    )
    def test_deep_module_import_names_flag(self, module_name: str) -> None:
        """Importing a deep module of an absent domain raises
        MissingDomainError, an ImportError, naming the flag to turn on."""
        (flag,) = _MODULE_FLAGS[module_name]
        with pytest.raises(MissingDomainError, match=flag) as info:
            importlib.import_module(module_name)
        assert isinstance(info.value, ImportError)

    @pytest.mark.parametrize(
        "name",
        [pytest.param(n, marks=_absent(f)) for n, f in _DOMAIN_NAMES.items()],
    )
    def test_top_level_name_names_flag(self, name: str) -> None:
        """A top-level name of an absent domain is a missing attribute whose
        message names the flag, so hasattr is False and star import skips it."""
        with pytest.raises(AttributeError, match=_DOMAIN_NAMES[name]):
            getattr(av, name)
        assert not hasattr(av, name)
        assert name not in av.__all__

    @_absent("ATV_BUILD_SEMIEMPIRICAL")
    def test_single_point_without_evaluate_names_flag(self) -> None:
        """driver imports on any build; single_point of a component evaluator
        on an extension without am1 raises TypeError naming its flag."""
        from atomiverse.driver import single_point

        with pytest.raises(TypeError, match="ATV_BUILD_SEMIEMPIRICAL"):
            single_point(None, object())
