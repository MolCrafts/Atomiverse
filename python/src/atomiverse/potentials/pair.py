"""Pair (non-bonded) potentials — ``cpu/potentials/`` pair styles."""

from atomiverse._binding import _cpp

#: Lennard-Jones 12-6 — ``cpu::PairLJCut(epsilon [eV], sigma [Å], cutoff [Å])``.
LJCut = _cpp.PairLJCut

__all__ = ["LJCut"]
