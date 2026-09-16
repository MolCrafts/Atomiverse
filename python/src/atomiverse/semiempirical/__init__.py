"""AM1 semiempirical electronic structure (mirrors ``cpu/semiempirical/``).

**Semiempirical** means a quantum-mechanical model of the valence electrons in
which the expensive electron-electron integrals are mostly dropped and the
survivors replaced by numbers fitted once per chemical element. The
approximation that does the dropping is NDDO — *neglect of diatomic
differential overlap* — and AM1, *Austin Model 1*, is one parameterization of
it. The result costs far less than a Gaussian-basis Hartree-Fock solve and far
more than a classical force field, and it is only as good as its fit.

``AM1Evaluator`` publishes energy [eV], Mulliken charges [e] and nuclear
forces [eV/Å], so it serves both single points and dynamics: hand it to
``atomiverse.driver.single_point`` for one solve, call ``forces(frame)`` for
the gradient, or register it with ``MD.set_evaluator`` for semiempirical
Born-Oppenheimer molecular dynamics (the electrons re-converged at every
geometry the nuclei reach). Gradients are a central finite difference of the
energy at fixed converged density (step 1e-4 Å), which is the exact NDDO
gradient up to the O(h²) truncation of that difference; closed-form NDDO
derivatives are a separate engine queue. Canonical deep paths:
``atomiverse.semiempirical.AM1`` / ``AM1Evaluator`` / ``AM1Result``; the
top-level names are the same classes.

Example::

    import atomiverse as atv

    ev = atv.AM1Evaluator(charge=0)   # or AM1Evaluator(atv.AM1(charge=0))
    atv.single_point(frame, ev)      # energy [eV] + charges [e] on frame
    res = ev.last_result()           # Hartree/kcal-mol detail + Mulliken
    f = ev.forces(frame)             # (N, 3) forces [eV/Å], also on fx/fy/fz

BOMD (Born-Oppenheimer molecular dynamics) is the same evaluator in the MD
evaluator slot. The frame needs a per-atom ``mass`` column [amu] here, which a
single point does not::

    sim = atv.MD()
    sim.set_integrator(atv.VelocityVerlet(0.25))   # [fs]
    sim.set_observer(atv.InitVelocity(300.0, 42))  # [K], RNG seed
    sim.set_evaluator(atv.AM1Evaluator())
    sim.run(frame, 100)

Neither ``last_result()`` nor the object handed to ``set_evaluator`` is
updated by an MD run — the driver installs its own gradient-enabled copy, and
the per-step energy, charges and forces land on the frame.
"""

from atomiverse._binding import _cpp

#: Concrete AM1 method — parameters assigned in the constructor
#: (``charge``, ``multiplicity``, ``max_iter``, ``density_tol``, ``energy_tol``).
AM1 = _cpp.AM1

#: The AM1 Evaluator component: energy, charges and forces from one solve.
AM1Evaluator = _cpp.AM1Evaluator

#: Converged-solve results: energies [eV / Hartree / kcal·mol⁻¹], Mulliken
#: charges [e], iteration count, and — after a force evaluation — the flat
#: nuclear-force buffer [eV/Å].
AM1Result = _cpp.AM1Result

#: SCC-DFTB2 for H/C/N/O — Elstner et al., Phys. Rev. B 58, 7260 (1998).
#:
#: A tight-binding valence Hamiltonian built from tabulated two-centre
#: integrals with the atomic charges relaxed self-consistently. Unlike AM1 it
#: has closed-form analytic forces and an analytic cell stress, and it is
#: periodic: `boundary` chooses between an isolated molecule and a 3-D lattice
#: with an Ewald-summed charge kernel. On the mio-1-1 parameter set it
#: reproduces DFTB+ to ~1e-8 eV/atom.
#:
#: Example::
#:
#:     d = atv.DFTB2()
#:     d.boundary = atv.DftbBoundary.lattice
#:     d.k_mesh = (3, 3, 3)
#:     forces = d.forces(frame)      # (N, 3) [eV/Å]; energy on frame.meta
#:
#: Open shell (triplet O₂, DFTB+ ``ShellResolvedSpin = No``)::
#:
#:     d = atv.DFTB2()
#:     d.boundary = atv.DftbBoundary.cluster
#:     d.multiplicity = 3
#:     d.spin_w_ha = atv.kDftbSpinConstantsPbeHa
DFTB2 = _cpp.DFTB2

#: Cluster vs lattice Hamiltonian — DFTB+'s `C` vs `S` geometry. These are
#: different Hamiltonians, not an approximation of one another: a large empty
#: box is NOT a way to get an isolated molecule once γ is Ewald-summed.
DftbBoundary = _cpp.DftbBoundary

#: Which transcribed Slater-Koster set DFTB2 / DFTB3 uses
#: (``unset``, ``mio``, or ``threeob``). DFTB2 defaults to ``mio``;
#: ``unset`` is the DFTB3 sentinel and throws from launch until assigned.
DftbSkSet = _cpp.DftbSkSet

#: One element's shell-resolved collinear spin constants \(W^A_{ll'}\) [Ha].
DftbSpinConstant = _cpp.DftbSpinConstant

#: Published PBE collinear spin-constant table [Ha] (Köhler / DFTB+ appendix).
#: Assign onto ``DFTB2.spin_w_ha`` for an open-shell solve; never applied
#: silently.
kDftbSpinConstantsPbeHa = _cpp.kDftbSpinConstantsPbeHa

#: LDA-functional collinear spin constants [Ha]. Not a default.
kDftbSpinConstantsLdaHa = _cpp.kDftbSpinConstantsLdaHa

__all__ = [
    "AM1",
    "AM1Evaluator",
    "AM1Result",
    "DFTB2",
    "DftbBoundary",
    "DftbSkSet",
    "DftbSpinConstant",
    "kDftbSpinConstantsPbeHa",
    "kDftbSpinConstantsLdaHa",
]
