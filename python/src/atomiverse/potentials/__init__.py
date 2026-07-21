"""Real-space force-field evaluators (mirrors ``cpu/potentials/``).

LAMMPS-style split: ``pair`` / ``bond`` / ``angle`` / ``dihedral``
submodules, prefix-stripped class names (``potentials.pair.LJCut`` is the
C++ ``cpu::PairLJCut``). The top-level convenience names keep the LAMMPS
compound spelling (``atomiverse.PairLJCut``) and are the same classes.
"""

from atomiverse.potentials import angle, bond, dihedral, pair

__all__ = ["pair", "bond", "angle", "dihedral"]
