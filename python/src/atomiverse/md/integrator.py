"""Time integrators (mirrors ``cpu/dynamics/`` integrator components).

An *integrator* is the component that advances positions and velocities by
one timestep. Exactly one is installed per run: the two NPT integrators
below fold thermostatting and barostatting into that same step, so they
*replace* ``VelocityVerlet`` rather than joining it. (NPT is the ensemble
at constant particle number, pressure and temperature.)

Canonical deep path: ``atomiverse.md.integrator.VelocityVerlet``; the
top-level convenience name ``atomiverse.VelocityVerlet`` is the same class.

Install via ``MD.set_integrator(VelocityVerlet(dt))``.
"""

from atomiverse._binding import _cpp

#: Symplectic velocity-Verlet integrator — ``cpu::md::VelocityVerlet(dt [fs])``.
VelocityVerlet = _cpp.VelocityVerlet

#: Martyna-Tobias-Klein (MTK) Nose-Hoover NPT integrator (LAMMPS ``fix
#: npt``) — ``cpu::md::NoseHooverNPT(T_target [K], tau_T [fs],
#: P_target [eV/Å³], tau_P [fs], dt [fs], drag = 0, mtchain = 3,
#: mpchain = 3, couple_axis = [True, True, True], T_stop = -1 [K])``.
#: The five leading arguments are required; the rest are optional LAMMPS
#: ``fix npt`` knobs (see the class docstring). Replaces the plain
#: integrator; the cell breathes along the enabled axes only.
NoseHooverNPT = _cpp.NoseHooverNPT

#: MTK integrator with full cell-matrix (Parrinello-Rahman) degrees of
#: freedom — ``cpu::md::ParrinelloRahmanNPT(T_target [K], P_target [eV/Å³],
#: tau_T [fs], tau_P [fs], dt [fs])``. All six strain-rate components
#: evolve, so the cell can shear as well as breathe. Replaces the plain
#: integrator. Mind the argument order: both setpoints first, then both
#: coupling times — the reverse pairing of ``NoseHooverNPT``.
ParrinelloRahmanNPT = _cpp.ParrinelloRahmanNPT

__all__ = ["VelocityVerlet", "NoseHooverNPT", "ParrinelloRahmanNPT"]
