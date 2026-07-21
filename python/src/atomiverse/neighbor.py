"""Neighbor structures (mirrors the ``cpu/neighbor/`` backend-root dir).

``neighbor/`` is a backend-root infrastructure directory in C++ — NOT part
of ``md/`` — so its Python home mirrors that: ``atomiverse.neighbor``.
Canonical deep path: ``atomiverse.neighbor.NeighborList``; top-level
``atomiverse.NeighborList`` is the same class.
"""

from atomiverse._binding import _cpp

#: Cell-list neighbor list — ``cpu::md::NeighborList<>(cutoff [Å], skin [Å])``.
NeighborList = _cpp.NeighborList

__all__ = ["NeighborList"]
