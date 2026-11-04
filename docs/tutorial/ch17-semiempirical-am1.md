# Semiempirical Models

Status tables: [Semiempirical roadmap](../roadmap/semiempirical.md).

A less expensive electronic model can be valuable when its domain is clear.
Semiempirical methods simplify parts of an electronic Hamiltonian and fit
selected terms to reference data. They save effort by making an explicit
trade between generality and cost, not by avoiding physical assumptions.

**Learning goals.** Explain the role of parameterization, distinguish energy
conventions, and use AM1 for a controlled exploratory geometry calculation.

**Before you begin.** The electronic mean-field picture in
[Hartree–Fock Theory](ch02-rhf.md).

## An approximate electronic Hamiltonian

AM1 belongs to the NDDO family. It uses a minimal valence description,
selected electronic interactions, and parameterized core terms. Its SCF
algebra resembles HF, but it solves a different approximate Hamiltonian.
Increasing a Gaussian basis is not a systematic AM1 improvement procedure.
Parameter availability also does not establish accuracy for every charge,
bonding pattern, or environment of that element.

A model total energy differs from a reported heat of formation. The latter
uses atomic reference quantities, schematically

$$\Delta H_f^{\mathrm{model}}=E_{\mathrm{mol}}^{\mathrm{model}}
-\sum_A E_A^{\mathrm{model}}+\sum_A\Delta H_{f,A}^{\mathrm{ref}}.$$

Record the convention before comparing programs. Neither value should be
subtracted indiscriminately from an ab initio energy with a different zero.

Forces must differentiate the actual parameterized energy. CPU and CUDA AM1/RM1
use analytic pair-integral derivatives within their supported element envelopes.
Electronic stationarity is essential: both the density and energy convergence
tolerances must be satisfied before forces are published.
On CPU the SCF diagonalizes a Pulay-DIIS extrapolation of the Fock matrix
(residual F P − P F, six vectors) while convergence is judged on the
unextrapolated density and energy changes, so DIIS changes the path, not the
converged answer. Once the commutator is below 0.05 eV, the full
eigensolver is replaced by a BLAS-3 pseudo-diagonalization (first-order
occupied–virtual rotation of the previous orbitals, re-orthonormalized),
again without changing the self-consistent result. Geometries with atoms closer than 0.6 × their covalent-radius
sum are refused before the SCF, naming the pairs.

## What the NDDO methods compute

| Quantity | Units | Where |
|----------|-------|--------|
| Total energy \(E_\mathrm{elec}+E_\mathrm{nuc}\) | eV | `frame.meta.get(kEnergy)` |
| Heat of formation \(\Delta H_f\) (MOPAC-style) | kcal/mol | `last_result().heat_of_formation_kcal_mol` |
| Mulliken-like charges | e | `kAtom_charge` |
| Nuclear forces | eV/Å | `kAtom_fx/fy/fz` when the phase requests forces |

Heat of formation follows the OpenMOPAC convention:

\[
\Delta H_f = (E_\mathrm{tot} - \sum_A e_\mathrm{isol}(A))\cdot c_{\mathrm{eV}\to\mathrm{kcal/mol}}
  + \sum_A E_\mathrm{atom}(A)
\]

with \(e_\mathrm{isol}\) from the selected method's parameter table,
\(E_\mathrm{atom}\) the experimental gas-phase atomic heats at 298 K, and
the repository's shared eV-to-kcal/mol conversion. Keep this convention and
the reference program's printed precision with any numerical comparison.

Forces use **three-direction analytic differentiation** of the same pair
integrals and core–core repulsion as the energy. The derivatives are contracted
with the converged density (P), without displaced geometries or differentiation
through the SCF iterations.

At electronic stationarity the orbital-response term vanishes. NDDO treats its
secular orbital basis as orthonormal, so no overlap Pulay term is needed.
Finite SCF tolerances still affect force accuracy; CPU AM1/RM1 require both
`density_tol` and `energy_tol` to be met. The method parameters are unchanged.

Each pair contributes equal-and-opposite forces, giving zero net force to
round-off. Independent CPU/CUDA full-vector checks use executed OpenMOPAC 23.2.5
outputs for distorted H₂O, tilted CO and NH₃, for both AM1 and RM1. The
oracle numerical-gradient error is qualified separately from Atomiverse.
These fixtures cover CHNO molecules; broader elements require their own
external evidence. The [fixture record](../../tools/tb_oracle/inputs/mopac/live-23.2.5/README.md)
contains the exact inputs, outputs and oracle gradient qualification. The
vibrational Hessian remains a finite difference of forces.

## Model selection

- Cost and accuracy depend on the system, parameter coverage and convergence.
  Measure complete calls at the required accuracy before choosing a backend.
- CPU AM1/RM1 mount directly in Evaluate, MD and Minimize
  ([Comparing Models](ch15-cross-method.md)). CUDA analytic forces and Evaluate
  are verified in the CHNO envelope. Both CUDA methods also pass short MD
  continuation and three-step steepest-descent checks. The examples below use CPU.
- For metals and covalent solids prefer many-body classical potentials
  ([Many-Body Potentials](ch16-many-body-potentials.md)) or GPW/PW DFT.

DFTB/xTB stay out of the NDDO family by design.

## Backend and method boundaries

CPU AM1/RM1 solve closed-shell molecules; open-shell requests throw. CPU retains
its existing parameter coverage and closed-shell charged calculations. RM1's
shipped table covers H/C/N/O/P/S; a parameter row is not evidence that every
chemical environment has been externally validated. The independent complete
force fixtures currently cover CHNO. A wrapping box can be used to unwrap one
isolated CPU fragment; it does not introduce periodic electronic interactions.
There is no NDDO lattice stress, solvent model or UHF implementation.

The C++ CUDA types are `cuda::semiempirical::AM1` and `RM1`. They provide total
energy, Mulliken charges, heat of formation and analytic forces for **neutral
closed-shell CHNO isolated molecules in FP64**. `run(frame)` writes energy and charges while
preserving existing forces. `run(const frame)` updates only the method result.
`forces(frame)` overwrites energy, charges and forces. An explicit
`last_result()` after a nested solve can download charges and requested forces.
Stress, non-CHNO species, nonzero charge, open shells and periodic cells are
rejected. CUDA Evaluate mounts the method directly and accumulates requested
energy/force contributions; its results and failure/retry behavior are checked
against standalone calls. Both CUDA methods pass initial-force preparation,
2+2 versus four-step MD continuation and CPU/CUDA four-step agreement at
`dt = 0.01 fs`. Three steepest-descent steps lower energy; this is not a
converged optimization result. Trial evaluations stage electronic history;
only a successful complete parent step accepts it. Between chunks, changes to
coordinates, velocities, elements or atom count invalidate affected state before
force preparation. Unsupported cell changes raise an error and can be corrected
on the same object. Consult the roadmap for method-specific numerical,
longer-NVE and performance evidence.

These methods implement NDDO scientific models using Atomiverse's own Driver,
workspace and derivative flow. They do not provide the whole SQM method set or
its input-file format. MNDO and PM3 use their own fitted CHNO parameter tables
with the same native NDDO integral, SCF and derivative machinery; their initial
public envelope is neutral closed-shell molecules on both backends. They do not
inherit AM1's broader element or charged-system support. PM6/PM7 and additional
SQM facilities remain separate work; acceptance is recorded per method in the
roadmap.

## Practice plan

Begin with neutral water at a slightly distorted geometry. Check electron
and charge counts, then compare force components with reconverged energy
differences. Zero net force is useful but not sufficient if translation
invariance has been enforced by construction.

Optimize with explicit force and displacement criteria. Compare bond lengths
and angle with another method, and re-evaluate both geometries using one
common reference Hamiltonian. This separates geometry changes from energy
convention changes. Semiempirical screening is most useful when the later
verification stage is planned from the beginning.

## Evaluator integration

```text
Preset (Evaluate / MD / Minimize / Hessian)
  └── AM1  :  NddoMethod = cpu::Driver<nddo::Phase>   (nested loop node)
         └── internal NDDO engine                    (Hamiltonian, density, forces)
```

The method itself is mounted in the outer loop; it runs its SCF on the
parent's Runtime. Mounted calls accumulate energy and requested forces, and
write the method's atomic charges. There is no evaluator wrapper. Successful
normal calls retain the accepted density as the next guess; initialization,
trial evaluations and failed solves do not advance that accepted history.

For standalone C++ calls, `run(Frame&)` overwrites energy and charges while
preserving forces; `forces(Frame&)` overwrites all three. `run(const Frame&)`
updates the method result without writing the Frame. First use builds the
workspace automatically; `build(reference)` is available for explicit setup.
`last_result()` exposes `ATV::semiempirical::NDDOResult` through the existing
method result alias. Numerical tolerances and iteration limits can be changed
between C++ solves; exhausting the budget raises an error rather than publishing
a successful result.

## Python method ownership

`AM1`, `RM1`, `MNDO` and `PM3` have the same Python call shape. Choose the
backend at construction; a mounted method retains its own controls and accepted
electronic state. The parent borrows that same native object.

```python
import atomiverse as av

# frame is an existing molecular Frame with coordinates, elements and masses.
method = av.AM1(device="cuda", max_iter=1000,
                density_tol=1e-10, energy_tol=1e-12)
force = method.forces(frame)  # owned (N, 3) array; also writes E/q/F to frame
method.run(frame)            # returns None; overwrites E/q, preserves frame forces
energy_ev = method.last_result().total_energy_ev

evaluation = av.Evaluate(device="cuda").set_evaluator(method)
evaluation.run(frame)
```

Python CUDA composition currently exposes VelocityVerlet MD and steepest
descent Minimize. The native C++ CUDA path also provides LBFGS. Unsupported
Python optimizer or MPI-launcher combinations reject explicitly. Continue a
Python CUDA MD run with the same Frame object; mutate its input columns between
chunks. Replacing that Python object is not a continuation operation.

## Example code

```cpp
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/core/stage.h"

cpu::Frame frame = /* water: type = Z, positions (Å) */;

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate)
    .add(cpu::semiempirical::AM1{});
eval.build();
eval.run(frame);

double E_eV = frame.meta.get(ATV::kEnergy);
// charges in kAtom_charge; forces in kAtom_fx/fy/fz
```

Like Gaussian SCF, bind it with `.on(phase).add(...)` (or `set_evaluator`).

## Geometry optimization

Same `Minimize` preset as classical or RHF — only the force component changes:

```cpp
cpu::minimizer::Minimize opt;
opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.02, 8});
opt.add_criterion(cpu::ForceConverged{/*max_f*/0.2, /*rms_f*/0.1,
                                      /*max_d*/0.1, /*rms_d*/0.05, /*dE*/1e-3});
opt.on(minimizer::Phase::force)
   .add(cpu::semiempirical::AM1{});
opt.build();
opt.run(frame, 50);
```

## Validation and performance evidence

Use the executed OpenMOPAC fixtures for reproducible numerical targets, rather
than mixing experimental heat of formation with a model total energy. The
AM1 regression checks energy, heat of formation, charges and complete CPU
force vectors for both AM1 and RM1 on CPU and CUDA, using the same six molecular
fixtures. The numerical-gradient
oracle's error budget is documented independently of Atomiverse's analytic
implementation.

The [standing force benchmark](../../benchmarks/semiempirical/nddo_forces/README.md)
records CPU complete-call timings and measured C++ allocation counts. The
[energy benchmark](../../benchmarks/semiempirical/nddo_energy/README.md) separates
CPU/CUDA host-Frame calls from nested resident CUDA solves. Benchmark source
alone establishes neither device residency nor a speedup; follow the current
validation status in the roadmap before relying on a backend.

## Questions for reflection

Which atomic reference terms cancel between isomers of the same composition? Why should an inexpensive optimized geometry be checked before using it in a different electronic method?
