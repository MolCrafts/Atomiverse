"""Time integrators (mirrors ``cpu/dynamics/`` integrator components).

Canonical deep path: ``atomiverse.md.integrator.VelocityVerlet``; the
top-level convenience name ``atomiverse.VelocityVerlet`` is the same class.

Install via ``MD.set_integrator(VelocityVerlet(dt))``.
"""

from atomiverse._binding import _cpp

#: Symplectic velocity-Verlet integrator — ``cpu::md::VelocityVerlet(dt [fs])``.
VelocityVerlet = _cpp.VelocityVerlet

#: MTK Nose-Hoover NPT integrator (LAMMPS ``fix npt``) —
#: ``cpu::md::NoseHooverNPT(T_target [K], tau_T [fs], P_target [eV/Å³],
#: tau_P [fs], dt [fs])``. Replaces the plain integrator.
NoseHooverNPT = _cpp.NoseHooverNPT

__all__ = ["VelocityVerlet", "NoseHooverNPT"]
