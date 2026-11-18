"""Bond (2-body bonded) potentials — ``cpu/potentials/`` bond styles."""

from atomiverse._binding import _cpp

#: Harmonic bond V = K(r - r0)^2 —
#: ``cpu::BondHarmonic(K [energy/length²] per type, r0 [length] per type)``.
Harmonic = _cpp.BondHarmonic

#: Kremer-Grest FENE bond U = −½KR₀²ln(1 − (r/R₀)²) —
#: ``cpu::BondFene(K [energy/length²] per type, R0 [length] per type)``. Finitely
#: extensible and loud about it: a bond at or beyond R₀ raises
#: ``RuntimeError`` from ``run()`` rather than reporting an infinite energy.
#: Normally paired with :data:`atomiverse.potentials.pair.WCA`, which is what
#: keeps polymer chains from crossing through each other.
Fene = _cpp.BondFene

__all__ = ["Harmonic", "Fene"]
