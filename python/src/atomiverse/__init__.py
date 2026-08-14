"""Atomiverse: molecular simulation compute backend for molrs.

Atomiverse binds drivers and components only — it does not bind a Frame
data type. Simulation state crosses the boundary as a ``molrs.Frame``;
each driver ``run()`` borrows it and writes results back in place. Global
results use the exact typed ``molrs.Frame.meta`` v2 contract.

Namespaces
----------
The DEEP paths are canonical — they mirror the C++ backend layout. The
top level re-exports the same objects under LAMMPS-style compound names
as a convenience (identity, not copies:
``atomiverse.PairLJCut is atomiverse.potentials.pair.LJCut``).

=====================================  ==========================  =========================
Canonical deep path                   Top-level re-export         C++ symbol
=====================================  ==========================  =========================
``md.MD``                             ``MD``                      ``cpu::md::MD``
``md.integrator.VelocityVerlet``      ``VelocityVerlet``          ``cpu::md::VelocityVerlet``
``md.integrator.NoseHooverNPT``       ``NoseHooverNPT``           ``cpu::md::NoseHooverNPT``
``md.integrator.ParrinelloRahmanNPT`` ``ParrinelloRahmanNPT``     ``cpu::md::ParrinelloRahmanNPT``
``md.thermostat.NoseHoover``          ``NoseHoover``              ``cpu::md::NoseHoover``
``md.thermostat.Langevin``            ``Langevin``                ``cpu::md::FixLangevin``
``md.thermostat.CSVR``                ``CSVR``                    ``cpu::md::FixCSVR``
``md.barostat.BerendsenBarostat``     ``BerendsenBarostat``       ``cpu::md::Barostat``
``md.InitVelocity``                   ``InitVelocity``            ``cpu::md::InitVelocity``
``minimizer.Minimize``                ``Minimize``                ``cpu::minimizer::Minimize``
``minimizer.LBFGS``                   ``LBFGS``                   ``cpu::minimizer::LBFGS``
``neighbor.NeighborList``             ``NeighborList``            ``cpu::md::NeighborList<>``
``potentials.pair.LJCut``             ``PairLJCut``               ``cpu::PairLJCut``
``potentials.bond.Harmonic``          ``BondHarmonic``            ``cpu::BondHarmonic``
``potentials.angle.Harmonic``         ``AngleHarmonic``           ``cpu::AngleHarmonic``
``potentials.dihedral.MultiHarmonic`` ``DihedralMultiHarmonic``   ``cpu::DihedralMultiHarmonic``
``recorder.Thermo``                   ``Thermo``                  ``cpu::Thermo``
``pw.PW``                             ``PW``                      ``cpu::pw::PW``
``pw.PWResult``                       ``PWResult``                ``cpu::pw::PWResult``
``semiempirical.AM1``                 ``AM1``                     ``cpu::semiempirical::AM1``
``semiempirical.AM1Evaluator``        ``AM1Evaluator``            ``cpu::semiempirical::AM1Evaluator``
``semiempirical.AM1Result``           ``AM1Result``               ``cpu::semiempirical::AM1Result``
``driver.single_point``               ``single_point``            ``cpu::single_point``
=====================================  ==========================  =========================

Drivers use the same **categorical setters** as C++ (not phase-only
``add`` as the primary path)::

    sim = av.MD()
    sim.set_evaluator(av.PairLJCut(0.0104, 3.4, 8.5))
    sim.set_neighbor(av.NeighborList(8.5, 0.5))
    sim.set_integrator(av.VelocityVerlet(1.0))
    sim.set_thermo(200)
    sim.build()
    sim.run(frame, 1000)

Backend selection is a ``device=`` constructor parameter (PyTorch-style),
never a namespace axis: ``MD(device="cpu")`` today; ``"cuda"`` raises
``ValueError`` until the CUDA bindings land.

A pure-Python wheel (no ``_atomiverse`` extension) still imports and
exposes ``__version__``; compute names raise an install-oriented
``ImportError`` pointing at ``python -m atomiverse config/compile/install``.
"""

from __future__ import annotations

from typing import Any

from atomiverse._binding import _cpp, extension_available, require_cpp

__version__ = "0.1.0"

# Top-level compute re-exports. Resolved lazily via PEP 562 so the package
# imports without the extension, and so submodules are not pulled in at
# import time (avoids stale editable maps and circular loads).
_COMPUTE_EXPORTS: dict[str, tuple[str, str]] = {
    "MD": ("atomiverse.md", "MD"),
    "InitVelocity": ("atomiverse.md", "InitVelocity"),
    "VelocityVerlet": ("atomiverse.md.integrator", "VelocityVerlet"),
    "NoseHooverNPT": ("atomiverse.md.integrator", "NoseHooverNPT"),
    "ParrinelloRahmanNPT": ("atomiverse.md.integrator", "ParrinelloRahmanNPT"),
    "NoseHoover": ("atomiverse.md.thermostat", "NoseHoover"),
    "Langevin": ("atomiverse.md.thermostat", "Langevin"),
    "CSVR": ("atomiverse.md.thermostat", "CSVR"),
    "BerendsenBarostat": ("atomiverse.md.barostat", "BerendsenBarostat"),
    "Minimize": ("atomiverse.minimizer", "Minimize"),
    "LBFGS": ("atomiverse.minimizer", "LBFGS"),
    "ForceConverged": ("atomiverse.minimizer", "ForceConverged"),
    "MinimizeResult": ("atomiverse.minimizer", "MinimizeResult"),
    "NeighborList": ("atomiverse.neighbor", "NeighborList"),
    "PairLJCut": ("atomiverse.potentials.pair", "LJCut"),
    "BondHarmonic": ("atomiverse.potentials.bond", "Harmonic"),
    "AngleHarmonic": ("atomiverse.potentials.angle", "Harmonic"),
    "DihedralMultiHarmonic": ("atomiverse.potentials.dihedral", "MultiHarmonic"),
    "Thermo": ("atomiverse.recorder", "Thermo"),
    "AM1": ("atomiverse.semiempirical", "AM1"),
    "AM1Evaluator": ("atomiverse.semiempirical", "AM1Evaluator"),
    "AM1Result": ("atomiverse.semiempirical", "AM1Result"),
    "single_point": ("atomiverse.driver", "single_point"),
    "PW": ("atomiverse.pw", "PW"),
    "PWResult": ("atomiverse.pw", "PWResult"),
}

__all__ = [
    "MD",
    "VelocityVerlet",
    "NoseHooverNPT",
    "ParrinelloRahmanNPT",
    "InitVelocity",
    "NoseHoover",
    "Langevin",
    "CSVR",
    "BerendsenBarostat",
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
    "AM1",
    "AM1Evaluator",
    "AM1Result",
    "single_point",
]


def __getattr__(name: str) -> Any:
    if name not in _COMPUTE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Extension must be present before any compute symbol resolves.
    require_cpp()
    mod_name, attr = _COMPUTE_EXPORTS[name]
    # PW types are compile-optional.
    if name in ("PW", "PWResult") and not hasattr(_cpp, "PW"):
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            "(plane-wave backend not built into this extension)"
        )
    import importlib

    mod = importlib.import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value  # cache
    return value


def __dir__() -> list[str]:
    names = list(globals().keys()) + list(_COMPUTE_EXPORTS.keys())
    if extension_available() and hasattr(_cpp, "PW"):
        pass  # PW names already in _COMPUTE_EXPORTS
    return sorted(set(names))
