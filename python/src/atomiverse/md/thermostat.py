"""Thermostats (mirrors ``cpu/md/`` thermostat components).

Canonical deep path: ``atomiverse.md.thermostat.NoseHoover``; the top-level
convenience name ``atomiverse.NoseHoover`` is the same class.
"""

from atomiverse._binding import _cpp

#: Nose-Hoover thermostat (LAMMPS ``fix nvt``) —
#: ``cpu::md::NoseHoover(T_target [K], tau_T [fs], dt [fs])``.
NoseHoover = _cpp.NoseHoover

__all__ = ["NoseHoover"]
