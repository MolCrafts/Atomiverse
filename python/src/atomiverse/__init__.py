"""Atomiverse: molecular simulation compute backend for molrs.

Atomiverse binds drivers and components only — it does not bind a Frame
data type. Simulation state crosses the boundary as a ``molrs.Frame``;
each driver ``run()`` borrows it and writes results back in place. Global
results use the exact typed ``molrs.Frame.meta`` v2 contract; Atomiverse does
not synthesize ``scalars`` or ``stress`` blocks.

Namespaces
----------
The DEEP paths are canonical — they mirror the C++ backend directory
layering one-to-one. The top level re-exports the same objects under
their LAMMPS-style compound names as a convenience (identity, not copies:
``atomiverse.PairLJCut is atomiverse.potentials.pair.LJCut``).

=====================================  ==========================  =========================
Canonical deep path                   Top-level re-export         C++ symbol (directory)
=====================================  ==========================  =========================
``md.MD``                             ``MD``                      ``cpu::md::MD`` (md/)
``md.integrator.VelocityVerlet``      ``VelocityVerlet``          ``cpu::md::VelocityVerlet`` (md/)
``md.thermostat.NoseHoover``          ``NoseHoover``              ``cpu::md::NoseHoover`` (md/)
``md.InitVelocity``                   ``InitVelocity``            ``cpu::md::InitVelocity`` (md/)
``minimizer.Minimize``                ``Minimize``                ``cpu::minimizer::Minimize`` (minimizer/)
``minimizer.LBFGS``                   ``LBFGS``                   ``cpu::minimizer::LBFGS`` (minimizer/)
``neighbor.NeighborList``             ``NeighborList``            ``cpu::md::NeighborList<>`` (neighbor/)
``potentials.pair.LJCut``             ``PairLJCut``               ``cpu::PairLJCut`` (potentials/)
``potentials.bond.Harmonic``          ``BondHarmonic``            ``cpu::BondHarmonic`` (potentials/)
``potentials.angle.Harmonic``         ``AngleHarmonic``           ``cpu::AngleHarmonic`` (potentials/)
``potentials.dihedral.MultiHarmonic`` ``DihedralMultiHarmonic``   ``cpu::DihedralMultiHarmonic`` (potentials/)
``recorder.Thermo``                   ``Thermo``                  ``cpu::Thermo`` (recorder/)
``pw.PWSolver`` / ``pw.PWConfig``     ``PWSolver`` / ``PWConfig`` ``cpu::pw`` (pw/)
``pw.PWResult``                       ``PWResult``                 ``cpu::pw`` (pw/)
``semiempirical.AM1Config``           ``AM1Config``               ``cpu::semiempirical`` (semiempirical/)
``semiempirical.AM1Evaluator``        ``AM1Evaluator``            ``cpu::semiempirical`` (semiempirical/)
``semiempirical.AM1Result``           ``AM1Result``               ``cpu::semiempirical`` (semiempirical/)
``driver.single_point``               ``single_point``            ``cpu::single_point`` (driver/)
=====================================  ==========================  =========================

Backend selection is a ``device=`` constructor parameter (PyTorch-style),
never a namespace axis: ``MD(device="cpu")`` today; ``"cuda"`` raises
``ValueError`` until the CUDA bindings land (their own spec).

A pure-Python wheel (no ``_atomiverse`` extension) still imports and
exposes ``__version__``; compute names raise an install-oriented
``ImportError`` pointing at ``python -m atomiverse install``.

Example — Molecular Dynamics (canonical deep paths)::

    import atomiverse as atv

    sim = atv.md.MD(dt=0.001)
    sim.add(atv.md.InitVelocity(10.0, 42))
    sim.on("force").add(atv.neighbor.NeighborList(8.5), order=-10) \\
                   .add(atv.potentials.pair.LJCut(0.0104, 3.4, 8.5))
    sim.run(frame, 100)
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
    "NoseHoover": ("atomiverse.md.thermostat", "NoseHoover"),
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
    "AM1Config": ("atomiverse.semiempirical", "AM1Config"),
    "AM1Evaluator": ("atomiverse.semiempirical", "AM1Evaluator"),
    "AM1Result": ("atomiverse.semiempirical", "AM1Result"),
    "single_point": ("atomiverse.driver", "single_point"),
    "PWConfig": ("atomiverse.pw", "PWConfig"),
    "PWSolver": ("atomiverse.pw", "PWSolver"),
    "PWResult": ("atomiverse.pw", "PWResult"),
}

__all__ = [
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
]


def __getattr__(name: str) -> Any:
    if name not in _COMPUTE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Extension must be present before any compute symbol resolves.
    require_cpp()
    mod_name, attr = _COMPUTE_EXPORTS[name]
    # PW types are compile-optional.
    if name in ("PWConfig", "PWSolver", "PWResult") and not hasattr(_cpp, "PWSolver"):
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
    if extension_available() and hasattr(_cpp, "PWSolver"):
        pass  # PW names already in _COMPUTE_EXPORTS
    return sorted(set(names))
