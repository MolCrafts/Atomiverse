# Chapter 17. Semiempirical AM1

Gaussian SCF and GPW give transferable electronic structure at a price.
Classical force fields are fast but parameterized. **Semiempirical** methods
sit in the middle: a minimal valence electronic Hamiltonian with element
parameters fitted once, then reused for organic chemistry at a cost between
classical MD and RHF.

Reference: MOPAC / openmopac AM1 \(\Delta H_f(\mathrm{H_2O})\approx -57.8\) kcal/mol.

## Architecture

```text
Preset (Evaluate / MD / Minimize / Hessian)
  └── AM1Evaluator  :  cpu::Semiempirical   (kind)
         └── AM1Solver : NDDOSolver         (inner SCF + forces)
```

There is **no** `SemiempiricalDriver`. DFTB and xTB, if added later, get their
own Solver + Evaluator + kind — not a seat on `NDDOSolver`.

## What AM1 computes

| Quantity | Units | Where |
|----------|-------|--------|
| Total energy \(E_\mathrm{elec}+E_\mathrm{nuc}\) | eV | `frame.meta.get(kEnergy)` |
| Heat of formation \(\Delta H_f\) (MOPAC-style) | kcal/mol | `AM1Result::heat_of_formation_kcal_mol` |
| Mulliken-like charges | e | `kAtom_charge` |
| Nuclear forces | eV/Å | `kAtom_fx/fy/fz` when `with_gradient(true)` |

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
Regressions live under `regressions/semiempirical/`.

Only \(3(N-1)\) differences are evaluated: the force on the last atom is set to
minus the sum of the others, exact because a rigid translation cannot change
\(E\). The forces therefore sum to zero by construction.

## Example code

```cpp
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/core/stage.h"

cpu::Frame frame = /* water: type = Z, positions (Å) */;

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate)
    .add(cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
eval.build();
eval.run(frame);

double E_eV = frame.meta.get(ATV::kEnergy);
// charges in kAtom_charge; forces in kAtom_fx/fy/fz
```

`default_bindings()` is empty on purpose: like SCF and PW, you **must** bind
with `.on(phase).add(...)`.

## Geometry optimization

Same `Minimize` preset as classical or RHF — only the force component changes:

```cpp
cpu::minimizer::Minimize opt;
opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.02, 8});
opt.add_criterion(cpu::ForceConverged{/*max_f*/0.2, /*rms_f*/0.1,
                                      /*max_d*/0.1, /*rms_d*/0.05, /*dE*/1e-3});
opt.on(minimizer::Phase::force)
   .add(cpu::semiempirical::AM1Evaluator{}.with_gradient(true));
opt.build();
opt.run(frame, 50);
```

## Live Results

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

## Where AM1 sits in the book

- Faster than RHF/STO-3G by roughly an order of magnitude for organics.
- Same Driver presets as everything else — only the force/energy component
  changes ([Chapter 15](ch15-cross-method.md)).
- For metals and covalent solids prefer many-body classical potentials
  ([Chapter 16](ch16-many-body-potentials.md)) or GPW/PW DFT.

DFTB/xTB stay out of the NDDO family by design.
