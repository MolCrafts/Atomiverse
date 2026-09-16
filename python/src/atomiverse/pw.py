"""Plane-wave DFT method API for periodic systems.

The PW-enabled wheel exposes the same object lifecycle as C++:

``PW`` (members) → ``build(frame)`` → ``run(frame)`` → ``PWResult``.

There is no bag named ``PWConfig`` / ``PWSolver`` and no transient functional
solve entry point.

Example::

    import atomiverse as av

    pw = av.PW()
    pw.gth_name = "GTH-PBE-q4"   # analytic GTH for Si; no external file
    pw.ecut_ry = 30.0
    pw.k_mesh = 4
    pw.build(frame)
    result = pw.run(frame)
    print(result.energy)  # Ry

A cell is not optional
----------------------

A plane-wave basis *is* the set of plane waves commensurate with the
simulation cell, selected by ``ecut_ry``. A frame with no ``box`` therefore
has no basis, and every entry point (``build``, ``run``, ``forces``,
``stress``) raises ``RuntimeError`` naming the box rather than iterating to a
NaN energy.

What you can ask for at a geometry
----------------------------------

============================  ==============================  ================
Call                          Returns                         Writes back
============================  ==============================  ================
``run(frame)``                ``PWResult`` (energies [Ry])    nothing
``forces(frame)``             ``(n_atoms, 3)`` [eV/Å]         fx/fy/fz, energy
``stress(frame)``             Voigt 6-vector [eV/Å³]          nothing
============================  ==============================  ================

``forces`` are analytic — local Hellmann-Feynman + Ewald ion-ion +
Kleinman-Bylander nonlocal, taken at the converged density rather than by
differencing energies — and the energy [eV] they land on
``frame.meta['energy']`` belongs to that same SCF. ``stress`` is the
Nielsen-Martin cell derivative in the mechanical (CP2K) sign,
σ = −(1/V)·∂E/∂ε, so a compressed cell gives a positive diagonal; it reports
on the cell and moves nothing. Both run their own SCF and leave
``last_result`` alone, which stays the result of the last ``run``.

Guards, all of them loud. A frame with no box raises ``RuntimeError``
everywhere. ``ValueError`` covers the physics preconditions: an odd
valence-electron count on the default restricted path, ``spin_polarized``
combined with a ``smearing`` (neither forces nor stress has that path yet),
a ``multiplicity`` whose parity contradicts the electron count, and a
``gth_name`` on a frame that mixes atomic numbers. Failing to converge
within ``max_iter`` is *not* an error — check ``result.converged``.

Units split at the boundary: ``PWResult`` keeps the solver's own Rydberg
(``energy``, ``e_fermi``, ``e_entropy``; 1 Ry = ½ Hartree, and the engine
spells 1 Ry = 13.605693 eV — the Quantum ESPRESSO-rounded
``cpu::pw::qe_units::rydberg_to_ev``) plus ``energy_ha`` in Hartree, while
everything written onto the frame is in engine units — eV, eV/Å, eV/Å³.

Knobs beyond the basis
----------------------

``xc``
    Exchange-correlation functional by *name* (default ``"PBE"``), the same
    table ``RKS(functional=...)`` uses: ``"LSDA"``, ``"PBE"``, ``"PBEsol"``,
    ``"RevPBE"``, ``"RPBE"``, ``"BLYP"``, ``"OptB88VdW"``, ``"B3LYP"``,
    ``"PBE0"``, ``"B2PLYP"``. An unknown name raises ``ValueError``.

``spin_polarized`` / ``multiplicity``
    Collinear spin. The default restricted path doubly occupies every band
    and so needs an even valence-electron count; an open-shell cell (a single
    H atom, say) sets ``spin_polarized = True`` and ``multiplicity = 2``,
    which fixes N↑ − N↓ = M − 1 exactly.

``smearing`` / ``degauss`` / ``mp_order`` / ``extra_bands``
    Metallic occupations by name — ``"none"`` (default, fixed occupations),
    ``"fermi_dirac"``, ``"methfessel_paxton"``, ``"gaussian"`` — of width
    ``degauss`` [Ry]. ``mp_order`` is read only by Methfessel-Paxton, and
    ``extra_bands`` is how many empty states above the Fermi level to solve
    for.

Plane-wave Born-Oppenheimer MD
------------------------------

``PWEvaluator`` is the same method wearing its Component face::

    from atomiverse.md import MD
    from atomiverse.md.integrator import VelocityVerlet
    from atomiverse.pw import PW, PWEvaluator

    pw = PW()
    pw.gth_name = "GTH-PBE-q4"
    pw.ecut_ry = 8.0
    pw.k_mesh = 1
    pw.max_iter = 12       # a step is two whole SCFs — budget it

    sim = MD()
    sim.set_integrator(VelocityVerlet(0.5))   # dt [fs]
    sim.set_evaluator(PWEvaluator(pw))
    sim.run(frame, 3)

Each step costs *two* SCF solves — the energy solve, then the analytic force
path's own — so the settings that make a Python-side run finish are a small
cell, a low ``ecut_ry`` and a modest ``max_iter``. The evaluator copies the
method's parameters (a ``PW`` owns FFTW plans, which cannot be shared), so
the object passed in stays usable and per-step results land on the frame
rather than on it.

Pseudopotential — exactly one of two, and there is no default
------------------------------------------------------------

``gth_name``
    An in-tree analytic Goedecker-Teter-Hutter parameterization, e.g.
    ``"GTH-PBE-q4"`` (Si) or ``"GTH-PBE-q1"`` (H). Needs no file: the
    published parameters ship with the engine. In-tree coverage is H, O,
    Na, Cl, Ar, Si, Ni, Zn, and one potential serves the whole cell, so
    the frame must be single-species.

``upf_path``
    A **norm-conserving** UPF file (ONCVPSP and similar). Ultrasoft and
    PAW files are rejected on read: the plane-wave path applies no
    augmentation charge, so integrating one would be wrong by an
    unbounded amount.

Either way the run is single-species: one potential and one valence-electron
count are applied to every atom in the cell. The ``gth_name`` branch says so
— a frame mixing atomic numbers raises ``ValueError`` — while the
``upf_path`` branch does not check, so a mixed-species frame there would
silently be treated as the UPF's own element throughout. Multi-species
plane-wave runs are not implemented.

Setting both, or neither, raises. There is no toy/bare-Coulomb fallback —
a bare valence-charge potential is not representable in a plane-wave basis
at any finite cutoff, so it never converged with ``ecut_ry``.

When Atomiverse is built without ``ATV_BUILD_PW=ON``, this module remains
importable but exports no PW types. The package surface therefore reflects the
compiled capability instead of pretending an unavailable solver exists.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

if hasattr(_cpp, "PW"):
    PW = _cpp.PW
    PWResult = _cpp.PWResult
    PWEvaluator = _cpp.PWEvaluator
    __all__ = ["PW", "PWResult", "PWEvaluator"]
else:
    __all__: list[str] = []
