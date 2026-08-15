"""Harmonic vibrational analysis (mirrors the ``cpu/vibration/`` backend dir).

What this computes
------------------
Near a stationary point — a geometry where every force vanishes — the
potential energy surface is well approximated by a quadratic form, i.e. by a
set of coupled springs. That is the **harmonic approximation**, and the
matrix of second derivatives of the energy with respect to the nuclear
Cartesian coordinates — the **Hessian** — is what turns a single geometry
into a spectrum. Diagonalizing it splits the coupled springs into
independent oscillations, the **normal modes**, each with one characteristic
frequency; those frequencies are reported as **wavenumbers** (a frequency
divided by the speed of light, the unit spectroscopy quotes, [cm⁻¹]).

``Hessian`` builds the matrix by CENTRAL FINITE DIFFERENCES — it moves each
coordinate a little either way and takes the difference of the forces,
rather than differentiating the energy expression analytically. Concretely
it displaces every Cartesian coordinate by ``±h`` (``h`` is fixed at 1e-3 Å),
differences the forces the registered evaluator produces, symmetrizes,
mass-weights, projects out the translations and rotations and diagonalizes
the rest. That is ``6N`` displaced force evaluations for ``N`` atoms (plus
one more at the restored geometry on the final pass) — the cost is the force
source's, so an electronic-structure evaluator on a large molecule is
expensive by construction.

=========================================  ==============================
``Hessian(...)``                           the finite-difference driver
``HessianResult``                          what ``run()`` returns
=========================================  ==============================

The mode count is the PROJECTED one
-----------------------------------
Six of the ``3N`` Cartesian directions (five if the molecule is linear) are
not vibrations at all: they are rigid translations and rotations of the
whole molecule, which cost no energy and so carry no restoring force. They
are projected out PySCF-style *before* the diagonalization, so
``result.frequencies`` has exactly ``3N-6`` entries — ``3N-5`` for a linear
molecule, and NOTHING at all (an empty array) for a lone atom, which has no
vibration to report. A diatomic reports ONE frequency. It is never "``3N``
values with five near-zero ones to filter"; there is nothing to filter.

An IMAGINARY mode — a direction in which the energy curves *down*, so the
geometry is a saddle point and not a minimum — comes back as a NEGATIVE
wavenumber (the universal ``imaginary_freq=False`` chemistry convention),
and ``n_imaginary`` counts them.

A force source is required
--------------------------
The force slot takes any bound energy/force source and has NO default: a run
with nothing registered raises ``RuntimeError``, because the Hessian of no
potential is not a physical zero. A pair style additionally needs a
``NeighborList``; an electronic-structure evaluator does not::

    import atomiverse as av

    driver = av.Hessian()
    driver.set_evaluator(av.PairLJCut(0.0104, 3.4, 8.5))
    driver.set_neighbor(av.NeighborList(8.5, 1.0))
    result = driver.run(frame)
    print(result.frequencies)          # [cm^-1]
    print(result.zpe, result.g_thermal)  # [Hartree]

Configure the driver fully before the first ``run()``: the execution plan is
compiled exactly once, so every setter raises ``RuntimeError`` afterwards,
and a different configuration means a new ``Hessian`` object.

The harmonic approximation only means anything at a stationary point, so
MINIMIZE FIRST (:mod:`atomiverse.minimizer`). Running on a non-stationary
geometry does not fail; it silently returns the curvature of a slope,
usually with spurious imaginary modes.

Thermochemistry: RRHO, and the σ you must supply
-------------------------------------------------
The same frequencies give the molecule's finite-temperature thermodynamics
under the **RRHO** (rigid-rotor / harmonic-oscillator) model: the molecule
translates and rotates as a rigid body while its normal modes oscillate as
independent quantum harmonic oscillators. Out of that come ``zpe`` — the
**zero-point energy**, the vibrational energy the molecule still has at 0 K
because a quantum oscillator cannot be brought to rest — and the
temperature-dependent internal energy, enthalpy, entropy and Gibbs free
energy (``u_tot``, ``h_tot``, ``s_*``, ``g_tot``, ``g_thermal``).

Four constructor arguments configure it, all defaulted::

    av.Hessian(temperature=298.15,   # [K]
               pressure=101325.0,    # [Pa] = 1 atm
               multiplicity=1,       # 2S+1; 1 = closed-shell singlet
               symmetry_number=1)    # rotational σ — SEE BELOW

``temperature`` scales every thermal term; ``pressure`` enters the
translational entropy and nothing else; ``multiplicity`` (the spin
multiplicity ``M = 2S+1``, the number of degenerate spin orientations of the
electronic ground state) contributes the electronic entropy ``R ln M`` and
nothing else.

``symmetry_number`` is NOT auto-detected — nothing inspects the molecule's
point group for you. σ is the number of indistinguishable orientations the
molecule has under rotation (2 for H₂O, 3 for NH₃, 12 for benzene); it
divides the rotational partition function, so leaving the default of 1 on a
symmetric molecule OVERESTIMATES the rotational entropy by ``R ln σ`` and
therefore under-estimates ``g_tot``. Supply it yourself.

One convention to know before comparing numbers with another code: the
electronic energy ``E0`` is carried as ZERO. ``u_tot``, ``h_tot`` and
``g_tot`` are therefore the THERMAL parts of U, H and G — add your own
electronic energy (in Hartree) for an absolute one — and ``g_tot`` and
``g_thermal`` come back holding the same number.

Masses come from Z, and Z ≤ 18
------------------------------
The analysis reads each atom's ``atomic_number`` column [Z] and looks the
IUPAC-2013 standard atomic weight up per element — the frame's per-atom
``mass`` column is IGNORED here, so editing it cannot move a frequency
(a mass mismatch shifts every wavenumber, ``ν ~ m^-1/2``, which is why the
lookup is not left to the caller). The table covers ``Z = 1..18`` (H..Ar)
and has no fallback: anything else raises ``ValueError`` naming the atomic
number.

No files unless you ask
-----------------------
``frequency_tsv`` and ``mode_xyz_prefix`` default to the EMPTY string, which
means "write nothing". Pass a path to opt in — a tab-separated frequency
report, and one ``<prefix>_mode<k>.xyz`` animation per real mode. The C++
preset defaults to ``frequencies.tsv`` / prefix ``mode`` instead; that is a
C++ caller's choice and deliberately not a Python one, since a binding that
littered the interpreter's working directory would do so on import-and-run.

Units
-----
Frequencies [cm⁻¹]; Hessian [eV/Å²]; energies (ZPE, U, H, G) [Hartree];
entropies [Hartree/K]; temperature [K]; pressure [Pa]; positions [Å];
forces [eV/Å]; masses [amu]. The thermochemistry is in atomic units because
it is a faithful port of PySCF's ``thermo()``; 1 Hartree = 27.211386 eV.
(The Hessian stays in the engine's own eV/Å² — the mass-weighting step
converts internally on its way to the frequencies.)

Scope
-----
This is MOLECULAR vibrational analysis — free-molecule translations and
rotations are projected out. Periodic phonons are a different calculation
and are not bound. Analytic (non-finite-difference) Hessians are likewise
not on this surface.

Vibrational analysis is CPU-only: there is no ``device=`` argument.
``ATV_BUILD_VIB`` is ON by default and the shipped wheel does not override
it, so both names stay in the top-level ``__all__``; only a staged CLI
install that leaves ``vib`` out drops them.
"""

from __future__ import annotations

from atomiverse._binding import _cpp

#: Central finite-difference Hessian driver — ``cpu::vib::Hessian``.
#: ``Hessian(temperature=298.15 [K], pressure=101325.0 [Pa],
#: multiplicity=1, symmetry_number=1, frequency_tsv="", mode_xyz_prefix="")``
#: — all defaulted; the two empty paths mean "write no files".
Hessian = _cpp.Hessian

#: Result of ``Hessian.run()`` — ``cpu::vib::HessianResult``: wavenumbers
#: [cm^-1], Cartesian Hessian [eV/A^2], normal modes, and rigid-rotor /
#: harmonic-oscillator (RRHO) thermochemistry [Hartree, Hartree/K] taken
#: relative to an electronic energy of zero.
HessianResult = _cpp.HessianResult

__all__ = ["Hessian", "HessianResult"]
