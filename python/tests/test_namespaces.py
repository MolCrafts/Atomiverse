"""Namespaced package layout tests (spec: python-api-namespaces-am1).

The atomiverse package mirrors the C++ backend directory layering as deep
canonical namespaces (``atv.potentials.pair.LJCut``,
``atv.md.integrator.VelocityVerlet``, ``atv.neighbor.NeighborList``, ...).
The deep path is the single source of truth; the top-level LAMMPS-style
composite names (``atv.PairLJCut``, ...) are identity re-exports of the
SAME class object — ``atv.PairLJCut is atv.potentials.pair.LJCut``.

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
from pathlib import Path

import numpy as np
import pytest
import molrs

import atomiverse as av
from atomiverse._binding import _cpp

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
    ("NeighborList", "atomiverse.neighbor", "NeighborList"),
    ("PairLJCut", "atomiverse.potentials.pair", "LJCut"),
    ("BondHarmonic", "atomiverse.potentials.bond", "Harmonic"),
    ("AngleHarmonic", "atomiverse.potentials.angle", "Harmonic"),
    ("DihedralMultiHarmonic", "atomiverse.potentials.dihedral", "MultiHarmonic"),
    ("Thermo", "atomiverse.recorder", "Thermo"),
    ("AM1Config", "atomiverse.semiempirical", "AM1Config"),
    ("AM1Evaluator", "atomiverse.semiempirical", "AM1Evaluator"),
    ("AM1Result", "atomiverse.semiempirical", "AM1Result"),
    ("single_point", "atomiverse.driver", "single_point"),
)

_PW_MAPPING: tuple[tuple[str, str, str], ...] = (
    (
        ("PWConfig", "atomiverse.pw", "PWConfig"),
        ("PWSolver", "atomiverse.pw", "PWSolver"),
        ("PWResult", "atomiverse.pw", "PWResult"),
    )
    if hasattr(_cpp, "PWSolver")
    else ()
)

_MAPPING = _BASE_MAPPING + _PW_MAPPING

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
    "atomiverse.recorder",
    "atomiverse.driver",
    "atomiverse.semiempirical",
    "atomiverse.pw",
)


# ── Deep modules importable ─────────────────────────────────────────


class TestDeepModulesImportable:
    @pytest.mark.parametrize("module_name", _DEEP_MODULES)
    def test_deep_module_imports(self, module_name: str) -> None:
        """Each deep canonical module imports on its own."""
        importlib.import_module(module_name)


# ── Top-level names are identity re-exports of the deep truth ───────


class TestTopLevelIdentity:
    def test_mapping_table_matches_compiled_surface(self) -> None:
        """The base has 18 entries and a PW build contributes three types."""
        assert len(_MAPPING) == (21 if hasattr(_cpp, "PWSolver") else 18)

    @pytest.mark.parametrize(
        "top_name,deep_module,deep_name",
        _MAPPING,
        ids=[row[0] for row in _MAPPING],
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


# ── The flat md.py implementation file is gone ──────────────────────


class TestFlatLayoutDeleted:
    def test_md_is_a_package(self) -> None:
        """atomiverse.md is a package (its __file__ is an __init__.py)."""
        md_mod = importlib.import_module("atomiverse.md")
        assert Path(md_mod.__file__).name == "__init__.py", (
            "atomiverse.md must be a package, not the flat md.py module"
        )

    def test_flat_md_file_does_not_exist(self) -> None:
        """python/src/atomiverse/md.py no longer exists as a file."""
        pkg_root = Path(av.__file__).resolve().parent
        assert not (pkg_root / "md.py").exists(), (
            "the flat implementation file atomiverse/md.py must be deleted "
            "(implementation moved into the md/ subpackage)"
        )


# ── device= parameter (constructor axis, v1 cpu-only) ───────────────


class TestDeviceParameter:
    def test_md_default_device_constructible(self) -> None:
        """MD(dt=...) with no device argument defaults to cpu."""
        from atomiverse.md import MD

        assert MD(dt=0.001) is not None

    def test_md_explicit_cpu_constructible(self) -> None:
        """MD(dt=..., device="cpu") is accepted."""
        from atomiverse.md import MD

        assert MD(dt=0.001, device="cpu") is not None

    def test_md_cpu_device_runs_lj_smoke(self, argon_frame: "molrs.Frame") -> None:
        """MD(device="cpu") runs the conftest LJ smoke to a finite energy."""
        from atomiverse.md import MD, InitVelocity
        from atomiverse.neighbor import NeighborList
        from atomiverse.potentials.pair import LJCut

        cutoff = 8.5
        sim = MD(dt=0.001, device="cpu")
        sim.add(InitVelocity(10.0, 42))
        sim.on("force").add(NeighborList(cutoff), order=-10).add(
            LJCut(0.0104, 3.4, cutoff)
        )
        sim.run(argon_frame, 5)
        energy = float(argon_frame.meta["energy"].value)
        assert np.isfinite(energy)

    def test_md_cuda_device_raises_value_error(self) -> None:
        """MD(device="cuda") raises ValueError naming CUDA."""
        from atomiverse.md import MD

        with pytest.raises(ValueError, match="(?i)cuda"):
            MD(dt=0.001, device="cuda")

    def test_minimize_cuda_device_raises_value_error(self) -> None:
        """Minimize(device="cuda") fails loud while GPU work is deferred."""
        from atomiverse.minimizer import Minimize

        with pytest.raises(ValueError, match="(?i)gpu|cuda"):
            Minimize(device="cuda")

    def test_single_point_cuda_device_raises_before_compute(
        self, argon_frame: "molrs.Frame"
    ) -> None:
        """single_point(..., device="cuda") raises ValueError BEFORE any
        compute: the argon frame (type=0) would itself fail AM1 evaluation
        with RuntimeError, so getting ValueError proves the device check
        precedes the calculation."""
        from atomiverse.driver import single_point
        from atomiverse.semiempirical import AM1Evaluator

        with pytest.raises(ValueError, match="(?i)cuda"):
            single_point(argon_frame, AM1Evaluator(), device="cuda")
