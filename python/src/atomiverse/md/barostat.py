"""Barostats (mirrors ``cpu/dynamics/`` barostat components).

A *barostat* holds a simulation at a target pressure by letting the
periodic cell change size: it reads the instantaneous pressure from the
stress tensor and rescales the box (and the atom positions with it).

Canonical deep path: ``atomiverse.md.barostat.BerendsenBarostat``; the
top-level convenience name ``atomiverse.BerendsenBarostat`` is the same
class.

Install via ``MD.set_barostat(BerendsenBarostat(P_target, tau_P, dt))``,
alongside an integrator and a thermostat — unlike ``NoseHooverNPT`` /
``ParrinelloRahmanNPT``, which replace the integrator instead.

Pressure is in eV/Å³ with the LAMMPS sign convention (positive means
compressive); **1 bar ≈ 6.2415e-7 eV/Å³**. The Berendsen scheme relaxes
the pressure smoothly but does *not* sample a rigorous NPT distribution —
use it to equilibrate a box, and an NPT integrator for production NPT
statistics.
"""

from atomiverse._binding import _cpp

#: Berendsen isotropic barostat (LAMMPS ``fix press/berendsen iso``) —
#: ``cpu::md::Barostat(P_target [eV/Å³], tau_P [fs], dt [fs],
#: bulk_modulus [eV/Å³] = 1.0)``. ``tau_P`` is the coupling time (how fast
#: the pressure is driven to its target) and ``bulk_modulus`` is the
#: tunable modulus β that divides the per-step rescale; the default 1.0 is
#: deliberately not LAMMPS's 10.0 (see ``cpu/dynamics/barostat.h``).
BerendsenBarostat = _cpp.BerendsenBarostat

__all__ = ["BerendsenBarostat"]
