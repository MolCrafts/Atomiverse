"""Evaluator registration at the Python/native boundary.

Parameter mirrors use the shared typed setter table, preserving their MPI
transport. Other bound Component and DriverNode objects use the native
registration protocol; the driver keeps their Python owners alive and refuses
MPI task dispatch when the task format cannot represent them.
"""

from __future__ import annotations

from atomiverse._binding import _cpp, has_domain

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
if has_domain("kspace"):
    EVALUATOR_DISPATCH[_cpp.KSpacePME] = "set_evaluator_kspace_pme"

# AM1 is the semiempirical force source (ATV_BUILD_SEMIEMPIRICAL): with it
# registered an MD loop is Born-Oppenheimer dynamics and a Minimize is a
# semiempirical geometry optimization. Capability-probed for the same reason
# as KSpacePME — the wheel builds it, a staged extension need not.
if has_domain("am1"):
    EVALUATOR_DISPATCH[_cpp.AM1] = "set_evaluator_am1"

# Plane-wave DFT is the other electronic-structure force source
# (ATV_BUILD_PW / FFTW3): one SCF per step, analytic ionic forces from its
# density, and the analytic virial whenever the plan reads stress (a cell
# relaxer or a stress/pressure criterion). Capability-probed like the two
# above.
if has_domain("pw"):
    EVALUATOR_DISPATCH[_cpp.PW] = "set_evaluator_pw"


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
        f"Supported: {supported}, native Component or DriverNode"
    )


def bind_evaluator(driver: object, component: object) -> None:
    """Register mirrored parameters or borrow a native component/loop node.

    Existing mirror entries retain their MPI transport behavior. Other live
    objects use the C++ registration protocol and remain owned by Python.
    """
    method = EVALUATOR_DISPATCH.get(type(component))
    if method is not None:
        getattr(driver, method)(component)
    elif isinstance(component, (_cpp._Component, _cpp._DriverNode)):
        driver.set_evaluator(component)
    else:
        raise TypeError(unsupported_evaluator_message(component))


__all__ = ["EVALUATOR_DISPATCH", "unsupported_evaluator_message", "bind_evaluator"]
