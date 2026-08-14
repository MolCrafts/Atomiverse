"""Long-range electrostatics — the two halves of one Ewald/PME split.

In a periodic cell the Coulomb sum ``sum q_i q_j / r_ij`` converges only
conditionally: ``1/r`` falls off too slowly to truncate at a cutoff. An
*Ewald split* fixes that by writing the sum as two absolutely convergent
pieces — a screened short-range part that does decay fast enough to
truncate, plus a smooth long-range part that converges in a handful of
Fourier components. The splitting parameter ``alpha`` [1/Å] sets where the
boundary between them falls:

``CoulLong``
    Real space — ``erfc(alpha·r) q_i q_j / r`` over the neighbor list
    (``cpu::CoulLong``, LAMMPS ``pair_style coul/long``). ``erfc`` is the
    complementary error function: 1 at ``r = 0``, decaying to 0 as
    ``alpha·r`` grows, so this half is the bare Coulomb interaction with a
    screen that switches it off before the cutoff.

``KSpacePME``
    Reciprocal space — smooth Particle-Mesh Ewald (Essmann et al., *J.
    Chem. Phys.* **103**, 8577 (1995)). Each charge is spread onto nearby
    mesh points with B-spline weights, the mesh is Fourier-transformed,
    multiplied by the *influence function* (the reciprocal-space kernel of
    the smooth term, fixed by the cell, ``alpha`` and ``order`` alone), and
    transformed back — O(N log N) instead of the O(N²) direct sum
    (``cpu::KSpacePME``).

Neither half is a Hamiltonian on its own. The driver enforces exactly one
matched real/reciprocal pair at ``build()``, so installing one without the
other raises there rather than running half an electrostatic model, and
both must be given the *same* ``alpha``::

    # LAMMPS rule of thumb: at alpha·cutoff = 3.5 the screened pair term
    # has fallen to erfc(3.5) ~ 7e-7 of the bare one by the cutoff.
    alpha = 3.5 / cutoff
    sim.set_evaluator(av.CoulLong(alpha, cutoff))
    sim.set_evaluator(av.KSpacePME(alpha, grid_spacing=0.5, order=6))

``CoulLong`` reads the neighbor list, so the run also needs a
``NeighborList`` built with a cutoff at least as large as the ``CoulLong``
cutoff. Larger ``alpha`` screens the real-space term harder and shifts work
into the reciprocal sum; smaller ``grid_spacing`` [Å] or larger ``order``
makes that reciprocal sum finer and costlier.

Charges are Frame data: every style here reads the per-atom ``charge``
column [e] of the ``molrs.Frame``, and ``run()`` rejects a frame that has
no such column instead of treating the system as neutral. A genuinely
neutral system is still expressible — insert an explicit zero column.

The reciprocal solver needs FFTW3 and is therefore compile-optional: on an
extension built without ``ATV_BUILD_KSPACE=ON`` this module still imports
and still exports ``CoulLong``, but ``KSpacePME`` is absent — the surface
reflects the compiled capability rather than pretending a solver exists.
For the cutoff (non-Ewald) Coulomb style see
:data:`atomiverse.potentials.pair.CoulCut`.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

#: Real-space Ewald/PME half — ``cpu::CoulLong(alpha [1/Å], cutoff [Å])``.
CoulLong = _cpp.CoulLong

if hasattr(_cpp, "KSpacePME"):
    #: Reciprocal-space Ewald/PME half —
    #: ``cpu::KSpacePME(alpha [1/Å], grid_spacing [Å] = 1.0, order = 6)``.
    KSpacePME = _cpp.KSpacePME
    __all__ = ["CoulLong", "KSpacePME"]
else:
    __all__ = ["CoulLong"]
