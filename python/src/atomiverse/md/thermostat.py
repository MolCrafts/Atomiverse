"""Thermostats (mirrors ``cpu/dynamics/`` thermostat components).

Canonical deep path: ``atomiverse.md.thermostat.NoseHoover``; the top-level
convenience name ``atomiverse.NoseHoover`` is the same class.

Install via ``MD.set_thermostat(NoseHoover(T, tau_T, dt))``.
"""

from atomiverse._binding import _cpp

#: Nose-Hoover thermostat (LAMMPS ``fix nvt``) —
#: ``cpu::md::NoseHoover(T_target [temperature], tau_T [time], dt [time])``.
NoseHoover = _cpp.NoseHoover

#: Langevin thermostat (LAMMPS ``fix langevin``) —
#: ``cpu::md::FixLangevin(T_start [temperature], T_stop [temperature], damp [time], dt [time], seed)``.
Langevin = _cpp.Langevin

#: Bussi canonical-sampling velocity rescale (CP2K CSVR, LAMMPS
#: ``fix temp/csvr``) on CPU and CUDA —
#: ``CSVR(T_start [temperature], tau [time], dt [time], seed, types=[],
#: bias="none")``. ``types`` lists the force-field types (molpy
#: ``atoms.type_id``) of the thermostatted group (empty: every atom);
#: ``bias="com"`` removes the group's centre-of-mass velocity before
#: measuring and rescaling (LAMMPS ``compute temp/com``). Several may be
#: set on one driver.
CSVR = _cpp.CSVR

#: Gauss least-constraint kinetic thermostat for ``GearPC`` —
#: ``GaussThermostat(kind, temperature [temperature], axis=0)`` with ``kind``
#: ``"kinetic"`` (barycentric), ``"flow"`` (species flow along ``axis``
#: excluded) or ``"transverse"`` (``axis`` excluded). Pass it to
#: ``GearPC(dt, thermostat)``, not to ``MD.set_thermostat``.
GaussThermostat = _cpp.GaussThermostat

#: Configurational Nose-Hoover thermostat for ``GearPC`` —
#: ``ConfigurationalThermostat(temperature [temperature], Q
#: [length^2/time^2])``. Pass it to ``GearPC(dt, thermostat)``.
ConfigurationalThermostat = _cpp.ConfigurationalThermostat

__all__ = [
    "NoseHoover",
    "Langevin",
    "CSVR",
    "GaussThermostat",
    "ConfigurationalThermostat",
]
