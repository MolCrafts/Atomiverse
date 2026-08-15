"""THE evaluator dispatch table — internal, one object, no copies.

An *evaluator* (or force source) is any component that, given a geometry,
produces an energy and the forces on the atoms: a Lennard-Jones pair style, a
bonded term, an Ewald electrostatic split, a semiempirical method, plane-wave
DFT. A force source is a force source — whatever produces energies and forces
for molecular dynamics produces them for a geometry optimization too — and the
C++ side says exactly that: ``Driver::set_evaluator`` is inherited, plural and
identical on ``cpu::md::MD`` and ``cpu::minimizer::Minimize``.

This module owns the one table that both Python facades import, so that
statement stays true here as well. Its keys are the compiled component classes
and its values are the typed ``set_evaluator_*`` method names the C++ handles
expose. Those names are deliberately spelled the same in ``bind_md.cpp`` and
``bind_minimizer.cpp``, which is what lets a single table drive both; a second
copy of it would be free to accept a component on one driver and refuse it on
the other, which is the failure this module exists to prevent.

Registration is unconditional for the components every build carries, and
capability-probed (``hasattr``) for the compile-optional ones, so the table
describes the extension actually installed rather than the maximal build.

Internal: nothing here is public API. Import it as
``from atomiverse._dispatch import EVALUATOR_DISPATCH as _EVALUATOR_DISPATCH``
so the facade attribute stays ``_EVALUATOR_DISPATCH`` and the object stays
shared by identity.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

#: Evaluator component type -> typed C++ setter name. Built once, at import,
#: and mutated only by the capability probes below.
EVALUATOR_DISPATCH: dict[type, str] = {
    _cpp.PairLJCut: "set_evaluator_pair_lj_cut",
    _cpp.PairCoulCut: "set_evaluator_pair_coul_cut",
    _cpp.CoulLong: "set_evaluator_coul_long",
    _cpp.BondHarmonic: "set_evaluator_bond_harmonic",
    _cpp.AngleHarmonic: "set_evaluator_angle_harmonic",
    _cpp.DihedralMultiHarmonic: "set_evaluator_dihedral_multi_harmonic",
    # The widened real-space catalogue: six pair styles and three bonded
    # ones. Unconditional like the six above — a real-space style is part of
    # ATV_BUILD_POTENTIALS with no build flag of its own, so no probe.
    _cpp.PairMorse: "set_evaluator_pair_morse",
    _cpp.PairBuck: "set_evaluator_pair_buck",
    _cpp.PairBorn: "set_evaluator_pair_born",
    _cpp.PairYukawa: "set_evaluator_pair_yukawa",
    _cpp.PairWCA: "set_evaluator_pair_wca",
    _cpp.PairDpd: "set_evaluator_pair_dpd",
    _cpp.BondFene: "set_evaluator_bond_fene",
    _cpp.DihedralCharmm: "set_evaluator_dihedral_charmm",
    _cpp.ImproperHarmonic: "set_evaluator_improper_harmonic",
}

# The PME reciprocal solver is compile-optional (ATV_BUILD_KSPACE / FFTW3),
# so it joins the allow-list only on an extension that carries it.
if hasattr(_cpp, "KSpacePME"):
    EVALUATOR_DISPATCH[_cpp.KSpacePME] = "set_evaluator_kspace_pme"

# AM1 is the semiempirical force source (ATV_BUILD_SEMIEMPIRICAL): with it
# registered an MD loop is Born-Oppenheimer dynamics and a Minimize is a
# semiempirical geometry optimization. Capability-probed for the same reason
# as KSpacePME — the wheel builds it, a staged extension need not.
if hasattr(_cpp, "AM1Evaluator"):
    EVALUATOR_DISPATCH[_cpp.AM1Evaluator] = "set_evaluator_am1"

# Plane-wave DFT is the other electronic-structure force source
# (ATV_BUILD_PW / FFTW3): analytic ionic forces from a re-converged density,
# two SCF solves per force-producing step. Capability-probed like the two
# above. Forces only — the setter installs a gradient-enabled copy of the
# method, not a stress-enabled one, so this source drives a geometry
# optimization but not a variable-cell relaxation.
if hasattr(_cpp, "PWEvaluator"):
    EVALUATOR_DISPATCH[_cpp.PWEvaluator] = "set_evaluator_pw"


def unsupported_evaluator_message(component: object) -> str:
    """Return the ``TypeError`` text for a component no facade accepts.

    One message for both facades, for the same reason as one table: the
    supported list must not go stale on one side.

    @param component: The object handed to ``set_evaluator``.
    @return Message naming the rejected type and every accepted one.
    """
    supported = ", ".join(t.__name__ for t in EVALUATOR_DISPATCH)
    return (
        f"set_evaluator does not support {type(component).__name__}. "
        f"Supported: {supported}"
    )


__all__ = ["EVALUATOR_DISPATCH", "unsupported_evaluator_message"]
