"""Bond (2-body bonded) potentials — ``cpu/potentials/`` bond styles."""

from atomiverse._binding import _cpp

#: Harmonic bond V = K(r - r0)^2 —
#: ``cpu::BondHarmonic(K [eV/Å²] per type, r0 [Å] per type)``.
Harmonic = _cpp.BondHarmonic

__all__ = ["Harmonic"]
