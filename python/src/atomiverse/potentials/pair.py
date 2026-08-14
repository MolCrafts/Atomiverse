"""Pair (non-bonded) potentials — ``cpu/potentials/`` pair styles."""

from atomiverse._binding import _cpp

#: Lennard-Jones 12-6 — ``cpu::PairLJCut(epsilon [eV], sigma [Å], cutoff [Å])``.
LJCut = _cpp.PairLJCut

#: Truncated Coulomb — ``cpu::PairCoulCut(cutoff [Å])``. The bare
#: ``q_i q_j / r`` sum over the neighbor list, hard-cut at ``cutoff``; needs
#: no reciprocal-space partner, so it is the whole electrostatic model of
#: the run that installs it. Needs a ``NeighborList`` built with a cutoff at
#: least as large. Charges come from the frame's per-atom ``charge``
#: column [e]; a frame without that column is rejected by ``run()`` rather
#: than read as neutral. The Ewald-split styles — accurate periodic
#: electrostatics, at the cost of a matched pair of components — live in
#: :mod:`atomiverse.longrange`.
CoulCut = _cpp.PairCoulCut

__all__ = ["LJCut", "CoulCut"]
