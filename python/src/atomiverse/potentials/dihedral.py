"""Dihedral (4-body bonded) potentials — ``cpu/potentials/`` dihedral styles."""

from atomiverse._binding import _cpp

#: Multi-harmonic (Ryckaert-Bellemans) dihedral V = Σₙ Cₙ cosⁿφ, φ being the
#: torsion angle about the central bond of the 4-tuple —
#: ``cpu::DihedralMultiHarmonic(Cn [eV] per dihedral type)``: one inner
#: sequence per type, its k-th entry multiplying cosᵏφ.
MultiHarmonic = _cpp.DihedralMultiHarmonic

#: CHARMM periodic torsion V = K(1 + cos(nφ − δ)) —
#: ``cpu::DihedralCharmm(K [eV] per type, n (int ≥ 1) per type,
#: delta [rad] per type)``. ``n`` is the number of minima per full turn, so
#: n = 3 is the threefold barrier of an sp3-sp3 bond and δ shifts where those
#: minima sit. This is LAMMPS ``dihedral_style charmm`` *without* the
#: optional 1-4 LJ/Coulomb weight factors — those belong to the pair styles
#: via special-bonds weights, not to the torsion.
Charmm = _cpp.DihedralCharmm

__all__ = ["MultiHarmonic", "Charmm"]
