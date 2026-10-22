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

Forces must differentiate the actual parameterized energy. A central
difference remains a numerical derivative with truncation and round-off
error, even if electronic stationarity eliminates explicit orbital response.
The preserved implementation discussion describes its own derivative route;
do not relabel it analytic without checking the code revision.

## What AM1 computes

| Quantity | Units | Where |
|----------|-------|--------|
| Total energy \(E_\mathrm{elec}+E_\mathrm{nuc}\) | eV | `frame.meta.get(kEnergy)` |
| Heat of formation \(\Delta H_f\) (MOPAC-style) | kcal/mol | `AM1Result::heat_of_formation_kcal_mol` |
| Mulliken-like charges | e | `kAtom_charge` |
| Nuclear forces | eV/Å | `kAtom_fx/fy/fz` when the phase requests forces |

Heat of formation follows the openmopac definition:

\[
\Delta H_f = (E_\mathrm{tot} - \sum_A e_\mathrm{isol}(A))\cdot 23.061
  + \sum_A E_\mathrm{atom}(A)
\]

with \(e_\mathrm{isol}\) from the AM1 parameter table and \(E_\mathrm{atom}\)
the experimental gas-phase atomic heats at 298 K.

Forces are a **central finite difference of the energy functional at fixed
density**: converge the SCF once, then displace one nucleus by \(\pm h\)
(\(h = 10^{-4}\) Å by default), rebuild the geometry-dependent integrals and
core–core repulsion with the density \(P\) unchanged, and difference the two
energies.

Freezing \(P\) costs no accuracy. At convergence the energy is stationary with
respect to density variations that conserve the electron count, so the response
term \((\partial E/\partial P)(dP/dR)\) vanishes; and NDDO's
zero-differential-overlap approximation makes the orbital basis orthonormal, so
there is no basis-derivative ("Pulay") term either. The approximation is the
difference quotient, which equals the exact NDDO gradient to \(O(h^{2})\) plus
round-off — which is why these forces match re-SCF energy finite differences.
Regressions live under `semiempirical`.

Only \(3(N-1)\) differences are evaluated: the force on the last atom is set to
minus the sum of the others, exact because a rigid translation cannot change
\(E\). The forces therefore sum to zero by construction.

## Model selection

- Faster than RHF/STO-3G by roughly an order of magnitude for organics.
- Same Driver presets as everything else — only the force/energy component
  changes ([Comparing Models](ch15-cross-method.md)).
- For metals and covalent solids prefer many-body classical potentials
  ([Many-Body Potentials](ch16-many-body-potentials.md)) or GPW/PW DFT.

DFTB/xTB stay out of the NDDO family by design.

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
parent's Runtime and accumulates energy, charges and — when the phase asks —
forces onto the Frame. There is no evaluator wrapper.

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

The following numerical material is retained from the source snapshot.
It was not rerun for this editorial revision; match its input and provenance
before using it as a validation target.

## Reference results

Pinned results:

| Quantity | Atomiverse | Reference |
|----------|-----------:|----------:|
| \(E\) (initial geometry) (eV) | **−348.295963** | — |
| \(\Delta H_f\) (kcal/mol) | **−53.093** | −57.8 (expt / MOPAC target) |
| \(q_\mathrm{O}\) (e) | **−0.3943** | O more negative than H |
| max\|F\| before Minimize (eV/Å) | 3.3135 | — |
| \(E\) after L-BFGS (eV) | **−348.562886** | lower |
| max\|F\| after Minimize (eV/Å) | **0.1513** | converged (`force_converged`) |

A ~5 kcal/mol error on water \(\Delta H_f\) is **typical for the AM1 model**,
not a bug — the method is parametrized for heats of formation across organics,
not exact on every molecule.

### Charge polarity

Oxygen carries the negative charge (\(q_O=-0.394\) e); hydrogens are positive
(\(q_H\approx +0.197\) e each by symmetry so \(\sum q_i=0\)). In MolVis, use a
darker colour for higher \(|q|\) when colouring by charge through the viewer
API after `molvis:ready`; the table above is the quantitative record.

## Questions for reflection

Which atomic reference terms cancel between isomers of the same composition? Why should an inexpensive optimized geometry be checked before using it in a different electronic method?
