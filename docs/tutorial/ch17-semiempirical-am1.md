# Chapter 17: Semiempirical AM1 — Cheap Quantum Forces

Gaussian SCF and GPW give transferable electronic structure at a price.
Classical force fields are fast but parameterized. **Semiempirical** methods
sit in the middle: a minimal valence electronic Hamiltonian with element
parameters fitted once, then reused for organic chemistry at a cost between
classical MD and RHF.

| | |
|---|---|
| **Example** | [`examples/15_am1.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/15_am1.cpp) |
| **Results** | [`examples/results/15_am1.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/15_am1.json) |
| **Reference** | MOPAC / openmopac AM1 \(\Delta H_f(\mathrm{H_2O})\approx -57.8\) kcal/mol |

```bash
cmake --build build --target 15_am1
./build/examples/15_am1
```

## Architecture

```text
Preset (Evaluate / MD / Minimize / Hessian)
  └── AM1Evaluator  :  cpu::Semiempirical   (kind)
         └── AM1Solver : NDDOSolver         (inner SCF + forces)
```

There is **no** `SemiempiricalDriver`. DFTB and xTB, if added later, get their
own Solver + Evaluator + kind — not a seat on `NDDOSolver`.

## The molecule: water

```molvis {format="xyz" representation="ball-and-stick" controls="view" height="300px"}
3
name=water Connct="[0,1,0,2]"
O  0.0000  0.0000  0.0000
H  0.9572  0.0000  0.0000
H -0.2390  0.9266  0.0000
```

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

Forces are the **SCF analytic gradient**: at a stationary density \(P\),
displace nuclei at fixed \(P\) and differentiate the energy functional
(Hellmann–Feynman for variational NDDO). They match re-SCF energy finite
differences; regressions live under `regressions/semiempirical/`.

## Example code

```cpp
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/facade/stage.h"

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

## Live results (`15_am1`)

From [`examples/results/15_am1.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/15_am1.json):

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
