"""Angle (3-body bonded) potentials — ``cpu/potentials/`` angle styles."""

from atomiverse._binding import _cpp

#: Harmonic angle V = K(theta - theta0)^2 —
#: ``cpu::AngleHarmonic(K [eV/rad²] per type, theta0 [rad] per type)``.
Harmonic = _cpp.AngleHarmonic

#: Cosine angle V = K(1 + cos theta) — LAMMPS ``angle_style cosine``,
#: ``cpu::AngleCosine(K [eV] per type)`` / ``cuda::AngleCosine``.
Cosine = _cpp.AngleCosine

__all__ = ["Harmonic", "Cosine"]
