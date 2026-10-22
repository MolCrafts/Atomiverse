"""Atomiverse: molecular simulation compute backend for molpy.

Atomiverse binds drivers and components only — it does not bind a Frame
data type. Simulation state crosses the boundary as a ``molpy.Frame``;
each driver ``run()`` borrows it and writes results back in place. Global
results use the exact typed ``molpy.Frame.meta`` v2 contract.

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
``minimizer.CG``                      ``CG``                      ``cpu::minimizer::CG``
``minimizer.FIRE``                    ``FIRE``                    ``cpu::minimizer::FIRE``
``minimizer.IsotropicCellRelax``      ``IsotropicCellRelax``      ``cpu::minimizer::IsotropicCellRelax``
``minimizer.AnisotropicCellRelax``    ``AnisotropicCellRelax``    ``cpu::minimizer::AnisotropicCellRelax``
``minimizer.TriclinicCellRelax``      ``TriclinicCellRelax``      ``cpu::minimizer::TriclinicCellRelax``
``minimizer.JointIsotropicRelax``     ``JointIsotropicRelax``     ``cpu::minimizer::JointIsotropicRelax``
``minimizer.ForceConverged``          ``ForceConverged``          ``cpu::ForceConverged``
``minimizer.PressureConverged``       ``PressureConverged``       ``cpu::PressureConverged``
``minimizer.StressConverged``         ``StressConverged``         ``cpu::StressConverged``
``minimizer.MinimizeResult``          ``MinimizeResult``          ``cpu::task::TaskResult``
``neighbor.NeighborList``             ``NeighborList``            ``cpu::md::NeighborList<>``
``potentials.pair.LJCut``             ``PairLJCut``               ``cpu::PairLJCut``
``potentials.pair.CoulCut``           ``PairCoulCut``             ``cpu::PairCoulCut``
``potentials.pair.Morse``             ``PairMorse``               ``cpu::PairMorse``
``potentials.pair.Buck``              ``PairBuck``                ``cpu::PairBuck``
``potentials.pair.Born``              ``PairBorn``                ``cpu::PairBorn``
``potentials.pair.Yukawa``            ``PairYukawa``              ``cpu::PairYukawa``
``potentials.pair.WCA``               ``PairWCA``                 ``cpu::PairWCA``
``potentials.pair.Dpd``               ``PairDpd``                 ``cpu::PairDpd``
``longrange.CoulLong``                ``CoulLong``                ``cpu::CoulLong``
``longrange.KSpacePME``               ``KSpacePME``               ``cpu::KSpacePME``
``potentials.bond.Harmonic``          ``BondHarmonic``            ``cpu::BondHarmonic``
``potentials.bond.Fene``              ``BondFene``                ``cpu::BondFene``
``potentials.angle.Harmonic``         ``AngleHarmonic``           ``cpu::AngleHarmonic``
``potentials.dihedral.MultiHarmonic`` ``DihedralMultiHarmonic``   ``cpu::DihedralMultiHarmonic``
``potentials.dihedral.Charmm``        ``DihedralCharmm``          ``cpu::DihedralCharmm``
``potentials.improper.Harmonic``      ``ImproperHarmonic``        ``cpu::ImproperHarmonic``
``recorder.Thermo``                   ``Thermo``                  ``cpu::Thermo``
``recorder.XYZ``                      ``XYZ``                     ``cpu::XyzRecorder``
``montecarlo.MC``                     ``MC``                      ``cpu::mc::MC``
``montecarlo.NptMC``                  ``NptMC``                   ``cpu::mc::NptMc``
``montecarlo.GCMC``                   ``GCMC``                    ``cpu::mc::Gcmc``
``montecarlo.TranslateMove``          ``TranslateMove``           ``cpu::mc::TranslateMove``
``montecarlo.VolumeMove``             ``VolumeMove``              ``cpu::mc::VolumeMove``
``montecarlo.InsertDeleteMove``       ``InsertDeleteMove``        ``cpu::mc::InsertDeleteMove``
``montecarlo.MoveSelector``           ``MoveSelector``            ``cpu::mc::MoveSelector``
``pw.PW``                             ``PW``                      ``cpu::pw::PW``
``pw.PWResult``                       ``PWResult``                ``cpu::pw::PWResult``
``scf.RHF``                           ``RHF``                     ``cpu::scf::RHF``
``scf.UHF``                           ``UHF``                     ``cpu::scf::UHF``
``scf.RKS``                           ``RKS``                     ``cpu::scf::RKS``
``scf.UKS``                           ``UKS``                     ``cpu::scf::UKS``
``semiempirical.AM1``                 ``AM1``                     ``cpu::semiempirical::AM1``
``semiempirical.AM1BCCCharges``       ``AM1BCCCharges``           ``cpu::semiempirical::AM1BCCCharges``
``semiempirical.BccParameterSet``     ``BccParameterSet``         ``cpu::semiempirical::BccParameterSet``
``semiempirical.AM1Result``           ``AM1Result``               ``cpu::semiempirical::AM1Result``
``semiempirical.DFTB2``               ``DFTB2``                   ``cpu::semiempirical::DFTB2``
``semiempirical.DftbBoundary``        ``DftbBoundary``            ``cpu::semiempirical::DftbBoundary``
``semiempirical.DftbSkSet``           ``DftbSkSet``               ``cpu::semiempirical::DftbSkSet``
``semiempirical.DftbSpinConstant``    ``DftbSpinConstant``        ``cpu::semiempirical::basis_data::DftbSpinConstant``
``semiempirical.kDftbSpinConstantsPbeHa`` ``kDftbSpinConstantsPbeHa`` ``cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa``
``vibration.Hessian``                 ``Hessian``                 ``cpu::vib::Hessian``
``vibration.HessianResult``           ``HessianResult``           ``cpu::vib::HessianResult``
``driver.single_point``               ``single_point``            ``cpu::single_point``
=====================================  ==========================  =========================

``RHF`` / ``UHF`` / ``RKS`` / ``UKS`` are the four Gaussian-basis SCF
(self-consistent field) methods — restricted / unrestricted Hartree-Fock and
Kohn-Sham DFT; ``help(atomiverse.scf)`` explains the naming and the units.

``MC`` / ``NptMC`` / ``GCMC`` are the three Monte Carlo ensemble presets:
canonical NVT (fixed particle number N, volume V, temperature T),
isothermal-isobaric NPT (fixed N, pressure P, T — the volume is sampled)
and grand-canonical muVT (fixed chemical potential mu, V, T — the particle
number is sampled). Monte Carlo samples configurations by proposing a
random trial move and accepting or rejecting it, instead of integrating
equations of motion, so a sampler takes a *mover* (``TranslateMove``,
``VolumeMove``, ``InsertDeleteMove``, or a weighted ``MoveSelector`` mix)
where MD takes an integrator, and never produces forces.
``help(atomiverse.montecarlo)`` gives the acceptance rules and the units.
These four movers and three presets are CPU-only and take no ``device=``.
An extension built without ``mc`` raises ``AttributeError`` naming the flag
for them, like the compile-optional rows below.

``Hessian`` / ``HessianResult`` are the harmonic vibrational analysis: the
central finite-difference Hessian over any registered force source, plus the
normal modes and the rigid-rotor/harmonic-oscillator (RRHO) thermochemistry
it diagonalizes into. Translations and rotations are projected out, so
``frequencies`` holds exactly ``3N-6`` entries (``3N-5`` when linear, none
at all for a lone atom) in cm⁻¹, and the thermochemistry is reported in
Hartree relative to an electronic energy of zero.
``help(atomiverse.vibration)`` gives the frame contract (masses come from
the ``atomic_number`` column, Z = 1..18), the rotational symmetry number you
must supply yourself, and the units. Like Monte Carlo it is CPU-only and takes
no ``device=``.

``LBFGS`` / ``CG`` / ``FIRE`` are the three fixed-cell geometry optimizers
and ``IsotropicCellRelax`` / ``AnisotropicCellRelax`` /
``TriclinicCellRelax`` / ``JointIsotropicRelax`` the four variable-cell
relaxers — they move the periodic *box* (isotropically, per axis, with shear,
or jointly with the atoms) towards a target pressure. The optimizer slot is
singular, so these seven are alternatives. Termination is a separate,
composable object: an optimizer publishes metrics and a criterion judges them,
which is ``ForceConverged`` for a fixed cell and ``PressureConverged`` /
``StressConverged`` for a variable one. A relaxer additionally needs a force
source that writes the cell virial, which the classical evaluators do and the
electronic-structure ones bound here do not; ``help(atomiverse.minimizer)``
gives that gate, the pairing rules and the units.

The real-space force-field catalogue is the ``set_evaluator`` slot's menu,
and the slot is plural: a full force field is several calls. Pair styles are
``PairLJCut`` (Lennard-Jones), ``PairMorse`` (a bonded-pair well),
``PairBuck`` / ``PairBorn`` (Born-Mayer repulsion, the latter as a per-type
Fumi-Tosi table with a +D/r⁸ core), ``PairYukawa`` (screened, chargeless),
``PairWCA`` (purely repulsive, cutoff derived from σ) and ``PairDpd``
(dissipative particle dynamics — a thermostat as well as a potential, so it
replaces rather than joins one). Bonded styles are ``BondHarmonic`` /
``BondFene``, ``AngleHarmonic``, ``DihedralMultiHarmonic`` /
``DihedralCharmm`` and ``ImproperHarmonic``, each reading the matching
topology block of the frame and indexing its parameters by that block's
type. Three practical notes. Among the *pair* styles only ``PairLJCut`` and
the electrostatic ones consume a ``NeighborList``; the other six walk all
pairs O(N²) and need none, so size the system accordingly (the bonded styles
walk topology tuples, not pairs, and never take a neighbor list either).
``PairBorn``'s type keys are ATOMIC NUMBERS, because that is what the frame's
per-atom type column carries. And ``PairDpd`` thermostats as well as
interacts, so it *replaces* ``NoseHoover`` / ``Langevin`` / ``CSVR`` instead
of joining them. All of them are unconditional — a real-space style has no
build flag of its own — and ``help(atomiverse.potentials)`` gives the deep
spellings and the units.

Every compute name belongs to one product module (``md``, ``minimize``,
``mc``, ``vib``, ``am1``, ``hf``, ``pw``, ``kspace``, ``potentials``). On an
extension built without that module, importing the deep module raises
``MissingDomainError`` (an ``ImportError``) and the top-level name raises
``AttributeError``, both naming the ``ATV_BUILD_*`` flag to turn on.

``__all__`` is the subset of the public names that the installed extension
actually carries: a native wheel holds only the modules it was configured
with, and ``from atomiverse import *`` must work on every such build.

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

from atomiverse._binding import (
    MissingDomainError,
    extension_available,
    has_domain,
    missing_domain_message,
    require_cpp,
)

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
    "CG": ("atomiverse.minimizer", "CG"),
    "FIRE": ("atomiverse.minimizer", "FIRE"),
    "IsotropicCellRelax": ("atomiverse.minimizer", "IsotropicCellRelax"),
    "AnisotropicCellRelax": ("atomiverse.minimizer", "AnisotropicCellRelax"),
    "TriclinicCellRelax": ("atomiverse.minimizer", "TriclinicCellRelax"),
    "JointIsotropicRelax": ("atomiverse.minimizer", "JointIsotropicRelax"),
    "ForceConverged": ("atomiverse.minimizer", "ForceConverged"),
    "PressureConverged": ("atomiverse.minimizer", "PressureConverged"),
    "StressConverged": ("atomiverse.minimizer", "StressConverged"),
    "MinimizeResult": ("atomiverse.minimizer", "MinimizeResult"),
    "NeighborList": ("atomiverse.neighbor", "NeighborList"),
    "PairLJCut": ("atomiverse.potentials.pair", "LJCut"),
    "PairCoulCut": ("atomiverse.potentials.pair", "CoulCut"),
    "PairMorse": ("atomiverse.potentials.pair", "Morse"),
    "PairBuck": ("atomiverse.potentials.pair", "Buck"),
    "PairBorn": ("atomiverse.potentials.pair", "Born"),
    "PairYukawa": ("atomiverse.potentials.pair", "Yukawa"),
    "PairWCA": ("atomiverse.potentials.pair", "WCA"),
    "PairDpd": ("atomiverse.potentials.pair", "Dpd"),
    "CoulLong": ("atomiverse.longrange", "CoulLong"),
    "KSpacePME": ("atomiverse.longrange", "KSpacePME"),
    "BondHarmonic": ("atomiverse.potentials.bond", "Harmonic"),
    "BondFene": ("atomiverse.potentials.bond", "Fene"),
    "AngleHarmonic": ("atomiverse.potentials.angle", "Harmonic"),
    "DihedralMultiHarmonic": ("atomiverse.potentials.dihedral", "MultiHarmonic"),
    "DihedralCharmm": ("atomiverse.potentials.dihedral", "Charmm"),
    "ImproperHarmonic": ("atomiverse.potentials.improper", "Harmonic"),
    "Thermo": ("atomiverse.recorder", "Thermo"),
    "XYZ": ("atomiverse.recorder", "XYZ"),
    "MC": ("atomiverse.montecarlo", "MC"),
    "NptMC": ("atomiverse.montecarlo", "NptMC"),
    "GCMC": ("atomiverse.montecarlo", "GCMC"),
    "TranslateMove": ("atomiverse.montecarlo", "TranslateMove"),
    "VolumeMove": ("atomiverse.montecarlo", "VolumeMove"),
    "InsertDeleteMove": ("atomiverse.montecarlo", "InsertDeleteMove"),
    "MoveSelector": ("atomiverse.montecarlo", "MoveSelector"),
    "AM1": ("atomiverse.semiempirical", "AM1"),
    "AM1BCCCharges": ("atomiverse.semiempirical", "AM1BCCCharges"),
    "BccParameterSet": ("atomiverse.semiempirical", "BccParameterSet"),
    "AM1Result": ("atomiverse.semiempirical", "AM1Result"),
    "DFTB2": ("atomiverse.semiempirical", "DFTB2"),
    "DftbBoundary": ("atomiverse.semiempirical", "DftbBoundary"),
    "DftbSkSet": ("atomiverse.semiempirical", "DftbSkSet"),
    "DftbSpinConstant": ("atomiverse.semiempirical", "DftbSpinConstant"),
    "kDftbSpinConstantsPbeHa": (
        "atomiverse.semiempirical",
        "kDftbSpinConstantsPbeHa",
    ),
    "kDftbSpinConstantsLdaHa": (
        "atomiverse.semiempirical",
        "kDftbSpinConstantsLdaHa",
    ),
    "Hessian": ("atomiverse.vibration", "Hessian"),
    "HessianResult": ("atomiverse.vibration", "HessianResult"),
    "single_point": ("atomiverse.driver", "single_point"),
    "PW": ("atomiverse.pw", "PW"),
    "PWResult": ("atomiverse.pw", "PWResult"),
    "RHF": ("atomiverse.scf", "RHF"),
    "UHF": ("atomiverse.scf", "UHF"),
    "RKS": ("atomiverse.scf", "RKS"),
    "UKS": ("atomiverse.scf", "UKS"),
}

# Names whose product module is not the one their deep module needs: the
# reciprocal solver lives in `longrange` (a `potentials` module) but is built
# by `kspace`, which pulls `potentials` in.
_NAME_DOMAINS: dict[str, str] = {"KSpacePME": "kspace"}

# The public top-level names. `__all__` is the subset the installed extension
# carries, computed on first access in `__getattr__`.
_PUBLIC: tuple[str, ...] = (
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
    "PairMorse",
    "PairBuck",
    "PairBorn",
    "PairYukawa",
    "PairWCA",
    "PairDpd",
    "CoulLong",
    "NeighborList",
    "BondHarmonic",
    "BondFene",
    "AngleHarmonic",
    "DihedralMultiHarmonic",
    "DihedralCharmm",
    "ImproperHarmonic",
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
    "AM1BCCCharges",
    "BccParameterSet",
    "AM1Result",
    "DFTB2",
    "DftbBoundary",
    "DftbSkSet",
    "DftbSpinConstant",
    "kDftbSpinConstantsPbeHa",
    "kDftbSpinConstantsLdaHa",
    "Hessian",
    "HessianResult",
    "single_point",
)


def _carried(name: str) -> bool:
    try:
        __getattr__(name)
    except (AttributeError, ImportError):
        return False
    return True


def __getattr__(name: str) -> Any:
    if name == "__all__":
        # Without the extension every name raises the install hint, which is
        # what a star import should surface; with it, keep what resolves.
        if not extension_available():
            return list(_PUBLIC)
        names = [n for n in _PUBLIC if _carried(n)]
        globals()["__all__"] = names
        return names
    if name not in _COMPUTE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Extension must be present before any compute symbol resolves.
    require_cpp()
    mod_name, attr = _COMPUTE_EXPORTS[name]
    domain = _NAME_DOMAINS.get(name)
    if domain is not None and not has_domain(domain):
        raise AttributeError(missing_domain_message(f"{__name__}.{name}", domain))
    import importlib

    try:
        mod = importlib.import_module(mod_name)
    except MissingDomainError as exc:
        # A name this build does not carry is a missing attribute, which is
        # what `hasattr` and `__all__` rely on; the message names the flag.
        raise AttributeError(
            missing_domain_message(f"{__name__}.{name}", exc.module)
        ) from exc
    value = getattr(mod, attr)
    globals()[name] = value  # cache
    return value


def __dir__() -> list[str]:
    names = list(globals().keys()) + list(_COMPUTE_EXPORTS.keys())
    return sorted(set(names))
