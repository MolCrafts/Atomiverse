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

#: Symplectic velocity-Verlet integrator — ``cpu::md::VelocityVerlet(dt [time])``.
VelocityVerlet = _cpp.VelocityVerlet

#: Martyna-Tobias-Klein (MTK) Nose-Hoover NPT integrator (LAMMPS ``fix
#: npt``) — ``cpu::md::NoseHooverNPT(T_target [temperature], tau_T [time],
#: P_target [pressure], tau_P [time], dt [time], drag = 0, mtchain = 3,
#: mpchain = 3, couple_axis = [True, True, True], T_stop = -1 [temperature])``.
#: The five leading arguments are required; the rest are optional LAMMPS
#: ``fix npt`` knobs (see the class docstring). Replaces the plain
#: integrator; the cell breathes along the enabled axes only.
NoseHooverNPT = _cpp.NoseHooverNPT

#: MTK integrator with full cell-matrix (Parrinello-Rahman) degrees of
#: freedom — ``cpu::md::ParrinelloRahmanNPT(T_target [temperature], P_target [pressure],
#: tau_T [time], tau_P [time], dt [time])``. All six strain-rate components
#: evolve, so the cell can shear as well as breathe. Replaces the plain
#: integrator. Mind the argument order: both setpoints first, then both
#: coupling times — the reverse pairing of ``NoseHooverNPT``.
ParrinelloRahmanNPT = _cpp.ParrinelloRahmanNPT

#: NVT SLLOD (LAMMPS ``fix nvt/sllod`` + ``fix deform`` ERATE, ``remap v``).
#: ``NvtSllod(temperature [temperature], tau [time], dt [time], rate [1/time], direction,
#: tchain=1)``. ``direction`` is 0..5 for x, y, z, yz, xz, xy; ``tchain`` is
#: the Nose-Hoover chain length (LAMMPS ``fix nvt/sllod`` default 1). A tilt
#: beyond half its box length flips the cell (LAMMPS ``flip yes``). Replaces
#: ``VelocityVerlet`` and includes the peculiar-velocity thermostat, so do not
#: add another thermostat. The MPI launcher cannot carry this integrator.
NvtSllod = _cpp.NvtSllod

__all__ = [
    "VelocityVerlet",
    "NoseHooverNPT",
    "ParrinelloRahmanNPT",
    "NvtSllod",
]
