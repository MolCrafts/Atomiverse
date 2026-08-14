"""Thermostats (mirrors ``cpu/dynamics/`` thermostat components).

Canonical deep path: ``atomiverse.md.thermostat.NoseHoover``; the top-level
convenience name ``atomiverse.NoseHoover`` is the same class.

Install via ``MD.set_thermostat(NoseHoover(T, tau_T, dt))``.
"""

from atomiverse._binding import _cpp

#: Nose-Hoover thermostat (LAMMPS ``fix nvt``) —
#: ``cpu::md::NoseHoover(T_target [K], tau_T [fs], dt [fs])``.
NoseHoover = _cpp.NoseHoover

#: Langevin thermostat (LAMMPS ``fix langevin``) —
#: ``cpu::md::FixLangevin(T_start [K], T_stop [K], damp [fs], dt [fs], seed)``.
Langevin = _cpp.Langevin

#: Bussi canonical-sampling velocity rescale (CP2K CSVR) —
#: ``cpu::md::FixCSVR(T_start [K], tau [fs], dt [fs], seed)``.
CSVR = _cpp.CSVR

__all__ = ["NoseHoover", "Langevin", "CSVR"]
