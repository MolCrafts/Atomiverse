"""Dihedral (4-body bonded) potentials — ``cpu/potentials/`` dihedral styles."""

from atomiverse._binding import _cpp

#: Multi-harmonic dihedral —
#: ``cpu::DihedralMultiHarmonic(Cn [eV] per dihedral type)``.
MultiHarmonic = _cpp.DihedralMultiHarmonic

__all__ = ["MultiHarmonic"]
