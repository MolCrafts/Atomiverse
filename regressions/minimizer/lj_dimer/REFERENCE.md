# Reference — LJ dimer geometry minimization

## Potential

Lennard-Jones pair (shifted form is irrelevant below the cutoff for this
isolated dimer):

```text
U(r) = 4 ε [(σ/r)¹² − (σ/r)⁶]
```

Analytic minimum:

```text
r* = 2^{1/6} σ
U(r*) = −ε
```

Force `F = −dU/dr` vanishes at `r*`. Units: ε [eV], σ [Å], force [eV/Å].

## Setup

Two atoms along x, initial separation `1.2 r*`, box `L = 40 Å` (≫ 2·cutoff)
so no periodic image interacts. Minimize with `PairLJCut` + `NeighborList` +
`LBFGS` + `ForceConverged`.

Parameters: `ε = 0.0104` eV, `σ = 3.4` Å, `cutoff = 2.5 σ`, same as classical
argon LJ style used elsewhere in Atomiverse.

## Tolerances

| Quantity | Tolerance | Derivation |
|---|---|---|
| \|r − r*\| | `0.01` Å | ≪ ForceConverged max-force threshold / curvature |
| \|U + ε\| | `1e-4` eV | well depth relative residual |
| max \|F\| | `0.05` eV/Å | ForceConverged `ftol` default order |
| stop reason | contains `force_converged` | handler name |

## Literature

Standard LJ dimer minimum from the derivative of the 12-6 form (any classical
MD textbook; e.g. Allen & Tildesley, *Computer Simulation of Liquids*).
