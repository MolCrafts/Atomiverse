"""Improper (4-body out-of-plane) potentials — ``cpu/potentials/``.

The fourth bonded category beside :mod:`~atomiverse.potentials.bond`,
:mod:`~atomiverse.potentials.angle` and
:mod:`~atomiverse.potentials.dihedral`. An *improper* is a 4-tuple that is
not a chain: it measures how far a centre has buckled out of the plane of
its three neighbours, which is what keeps an sp2 centre or an aromatic ring
flat. It travels in its own ``impropers`` block of the frame, separate from
``dihedrals``.
"""

from atomiverse._binding import _cpp

#: Harmonic improper V = K(chi - chi0)^2 —
#: ``cpu::ImproperHarmonic(K [eV/rad²] per type, chi0 [rad] per type)``,
#: where chi is the angle between the two planes of the 4-tuple (LAMMPS
#: ``improper_style harmonic``).
Harmonic = _cpp.ImproperHarmonic

#: Amber/GAFF improper V = K(1 + d cos(n phi)), d = ±1 —
#: ``cpu::ImproperCvff(K [eV] per type, d per type, n per type)`` (LAMMPS
#: ``improper_style cvff``). phi is the torsion angle of the improper 4-tuple
#: in its frame order; Amber puts the central atom third. Local runs only —
#: not carried by the MPI task wire.
Cvff = _cpp.ImproperCvff

__all__ = ["Harmonic", "Cvff"]
