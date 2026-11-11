"""Semiempirical electronic structure with native CPU/CUDA method selection.

**Semiempirical** means a quantum-mechanical model of the valence electrons in
which the expensive electron-electron integrals are mostly dropped and the
survivors replaced by numbers fitted once per chemical element. The
approximation that does the dropping is NDDO — *neglect of diatomic
differential overlap* — and AM1, *Austin Model 1*, is one parameterization of
it. The result costs far less than a Gaussian-basis Hartree-Fock solve and far
more than a classical force field, and it is only as good as its fit.

``AM1`` publishes energy [eV], Mulliken charges [e] and nuclear forces
[eV/Å], so it serves both single points and dynamics: hand it to
``atomiverse.driver.single_point`` for one solve, call ``forces(frame)`` for
the forces, or register it with ``MD.set_evaluator`` for semiempirical
Born-Oppenheimer molecular dynamics (the electrons re-converged at every
geometry the nuclei reach). CPU AM1/RM1 forces use three-direction analytic
Cartesian derivatives of the pair integrals and core repulsion, contracted
with the converged density. The model parameters are unchanged. Electronic
stationarity is required: both density and energy tolerances must be met.
The vibrational Hessian remains a finite difference of forces. Canonical deep paths:
``atomiverse.semiempirical.AM1`` / ``AM1Result`` and corresponding RM1,
MNDO and PM3 constructors/results; the top-level names are identity exports.

Example::

    import atomiverse as atv

    am1 = atv.AM1(charge=0)
    atv.single_point(frame, am1)     # energy [eV] + charges [e] on frame
    res = am1.last_result()          # Hartree/kcal-mol detail + Mulliken
    f = am1.forces(frame)            # (N, 3) forces [eV/Å], also on fx/fy/fz

BOMD (Born-Oppenheimer molecular dynamics) is the same method in the MD
evaluator slot. The frame needs a per-atom ``mass`` column [amu] here, which a
single point does not::

    sim = atv.MD()
    sim.set_integrator(atv.VelocityVerlet(0.25))   # [fs]
    sim.set_observer(atv.InitVelocity(300.0, 42))  # [K], RNG seed
    sim.set_evaluator(atv.AM1())
    sim.run(frame, 100)

Drivers borrow the method and keep its Python owner alive. Results remain
on the original object. ``run(frame)`` updates energy/charges while preserving
Frame forces; ``forces(frame)`` also overwrites and returns analytic forces.
Use ``device="cuda"`` for the supported native CUDA molecular subset.
"""

from atomiverse._binding import _cpp, require_domain, backend_type

require_domain(__name__, "am1")

#: Concrete AM1 method — parameters assigned in the constructor
#: (``charge``, ``multiplicity``, ``max_iter``, ``density_tol``, ``energy_tol``).
def AM1(charge: int = 0, multiplicity: int = 1, max_iter: int = 200,
        density_tol: float = 1e-8, energy_tol: float = 1e-10, *, device: str = "cpu"):
    """Construct native AM1; CUDA currently accepts neutral closed-shell CHNO."""
    return backend_type(device, "AM1", "_CudaAM1")(
        charge, multiplicity, max_iter, density_tol, energy_tol)


def RM1(charge: int = 0, multiplicity: int = 1, max_iter: int = 200,
        density_tol: float = 1e-8, energy_tol: float = 1e-10, *, device: str = "cpu"):
    """Construct native RM1 with the same Frame/results contract as AM1."""
    return backend_type(device, "RM1", "_CudaRM1")(
        charge, multiplicity, max_iter, density_tol, energy_tol)


def MNDO(charge: int = 0, multiplicity: int = 1, max_iter: int = 200,
         density_tol: float = 1e-8, energy_tol: float = 1e-10, *, device: str = "cpu"):
    """Construct native MNDO; CUDA supports neutral closed-shell CHNO."""
    return backend_type(device, "MNDO", "_CudaMNDO")(
        charge, multiplicity, max_iter, density_tol, energy_tol)


def PM3(charge: int = 0, multiplicity: int = 1, max_iter: int = 200,
        density_tol: float = 1e-8, energy_tol: float = 1e-10, *, device: str = "cpu"):
    """Construct native PM3 with the same Frame/results contract as MNDO."""
    return backend_type(device, "PM3", "_CudaPM3")(
        charge, multiplicity, max_iter, density_tol, energy_tol)


#: Converged-solve results: energies [eV / Hartree / kcal·mol⁻¹], Mulliken
#: charges [e], iteration count, and — after a force evaluation — the flat
#: nuclear-force buffer [eV/Å].
AM1Result = _cpp.AM1Result
RM1Result = _cpp.RM1Result
MNDOResult = _cpp.MNDOResult
PM3Result = _cpp.PM3Result

#: AM1-BCC / ABCG2 partial charges — the GAFF / GAFF2 charge model, as
#: antechamber ``-c bcc`` / ``-c abcg2`` compute them, at the frame's
#: geometry. The molecule is a molpy.Atomistic chemical graph::
#:
#:     model = atv.AM1BCCCharges(method=atv.AM1(charge=-1), correction="bcc")
#:     charges = model.assign(molecule)  # owned float64 charges [e], graph unchanged
AM1BCCCharges = _cpp.AM1BCCCharges


#: Non-SCC mio-1-1 energy, charges and forces for neutral closed-shell CHNO.
DFTB0 = _cpp.DFTB0

#: SCC-DFTB2 for H/C/N/O — Elstner et al., Phys. Rev. B 58, 7260 (1998).
#:
#: A tight-binding valence Hamiltonian built from tabulated two-centre
#: integrals with the atomic charges relaxed self-consistently. Like AM1 it
#: provides analytic forces; it also provides analytic cell stress and is
#: periodic: `boundary` chooses between an isolated molecule and a 3-D lattice
#: with an Ewald-summed charge kernel. On the mio-1-1 parameter set it
#: reproduces DFTB+ to ~1e-8 eV/atom.
#:
#: Example::
#:
#:     d = atv.DFTB2(atv.DftbSkSet.mio)
#:     d.boundary = atv.DftbBoundary.lattice
#:     d.k_mesh = (3, 3, 3)
#:     forces = d.forces(frame)      # (N, 3) [eV/Å]; energy on frame.meta
#:
#: Open shell (triplet O₂, DFTB+ ``ShellResolvedSpin = No``)::
#:
#:     d = atv.DFTB2(atv.DftbSkSet.mio, multiplicity=3)
#:     d.boundary = atv.DftbBoundary.cluster
#:     d.spin_w_ha = atv.kDftbSpinConstantsPbeHa
DFTB2 = _cpp.DFTB2

#: Cluster vs lattice Hamiltonian — DFTB+'s `C` vs `S` geometry. These are
#: different Hamiltonians, not an approximation of one another: a large empty
#: box is NOT a way to get an isolated molecule once γ is Ewald-summed.
DftbBoundary = _cpp.DftbBoundary

#: Which transcribed Slater-Koster set DFTB2 uses (``unset``, ``mio``, or
#: ``threeob``) — the required first argument of ``DFTB2(sk_set, charge=0,
#: multiplicity=1)``. ``unset`` is a sentinel that throws from launch.
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
    "RM1",
    "MNDO",
    "PM3",
    "AM1BCCCharges",
    "AM1Result",
    "RM1Result",
    "MNDOResult",
    "PM3Result",
    "DFTB0",
    "DFTB2",
    "DftbBoundary",
    "DftbSkSet",
    "DftbSpinConstant",
    "kDftbSpinConstantsPbeHa",
    "kDftbSpinConstantsLdaHa",
]
