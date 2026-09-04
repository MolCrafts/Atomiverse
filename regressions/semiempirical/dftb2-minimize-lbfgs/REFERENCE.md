# dftb2-minimize-lbfgs — L-BFGS on SCC-DFTB2 forces to fmax = 0.05 eV/Å

```text
oracle: literature
oracle_version: protocol threshold (Hooven et al. 2026); internal gate
family_oracle: dftb+
```

## Why no DFTB+ geometry pin

Two L-BFGS implementations with different line searches reach different points
in the same basin, so pinning ATV's relaxed geometry to DFTB+'s would pin the
optimiser, not the physics. The physics on this path (energy, analytic forces
under Ewald γ) is pinned in `dftb2-ewald-gamma`. What is gated here is that the
loop converges by its own criterion and that **the criterion means what it
says**.

## The threshold

`fmax = 0.05 eV/Å` is not a round number picked for the test: it is the
convergence gate the DFTB+ protocol in Hooven, Lin, Carroll, Cersonsky,
*J. Chem. Theory Comput.* **22**, 6917 (2026) specifies for its n=8 relaxation,
`GeometryOptimisation { Optimiser = LBFGS {}; Convergence = { GradElem
[eV/angstrom] = 0.05 } }`.

## Method

`cpu::minimizer::Minimize` + `cpu::minimizer::LBFGS{5, 0.02, 10}` +
`cpu::ForceConverged{0.05, 0.05, 1.0, 1.0, 1e-2}` on `DFTB2` with
`boundary = lattice`, so the Ewald γ gradient is on the optimisation path and
not only on a single-point one. The displacement and energy legs of
`ForceConverged` are left loose on purpose: all five of its criteria must hold
simultaneously, and this case is about the **force** leg, not about whichever
leg happens to trip first.

## System

2 waters in a periodic cubic 6.0 Å cell with bond lengths and angles pushed off
equilibrium (start max\|F\| = 3.89 eV/Å, i.e. well above the gate — a case that
started converged would gate nothing).

## Gates

| Gate | Requirement | Observed |
| ---- | ----------- | -------- |
| Start is off-equilibrium | max\|F\|₀ > 0.05 eV/Å | **3.89 eV/Å** |
| Loop converges | `converged()` within 60 iterations | yes, `force_converged` |
| **Criterion means what it says** | re-measured max\|F\| ≤ 0.05 eV/Å | **0.0373 eV/Å** |
| Energy decreases | E₁ < E₀ | −221.205 → **−222.186 eV** |

The third gate re-evaluates the forces with a fresh `cpu::Evaluate` rather than
trusting the optimiser's own last internal evaluation, and measures the max
force **vector magnitude** — which is what `ForceConverged` judges
(`src/cpu/minimize/lbfgs.cpp` publishes `sqrt(fx²+fy²+fz²)`, not the max
Cartesian component).

## Re-run

```bash
cmake --build <build> --target run_reg_dftb2-minimize-lbfgs
```
