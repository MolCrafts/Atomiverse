# FMM point-charge regression — literature anchors

## What this regression checks

Open-boundary Coulomb energy of point-charge sets via
`ATV::longrange::FmmTree`, compared to the O(N²) direct sum
Σ_{i<j} q_i q_j / r_ij (dimensionless kernel — no `units::coulomb`).

1. **Order convergence.** Relative energy error vs direct sum for random and
   simple-cubic lattice charge sets, p = 2,4,…p★. The every-two-order error
   envelope must decrease geometrically (ratio < 1 while above roundoff). At
   the documented order **p★ = 16**, relative error ≤ **1e-10**.
2. **Near-linear scaling smoke test.** Wall time for N and 4N particles
   (same p, same leaf capacity): t(4N)/t(N) < 8 (well below the O(N²)
   expectation of 16).

## Citations

| Claim | Source |
|---|---|
| FMM algorithm, geometric error bound ∼ (1/(c−1))(1/c)^p | Greengard & Rokhlin, *J. Comput. Phys.* **73**, 325 (1987), DOI [10.1016/0021-9991(87)90140-9](https://doi.org/10.1016/0021-9991(87)90140-9) |
| Real solid-harmonic M2M/M2L/L2L | White & Head-Gordon, *J. Chem. Phys.* **101**, 6593 (1994), DOI [10.1063/1.468354](https://doi.org/10.1063/1.468354) |
| Linear-scaling Coulomb motivation | F. Jensen, *Introduction to Computational Chemistry*, SCF-bottleneck / linear-scaling section |

## Documented order p★

| Quantity | Value |
|---|---|
| p★ (regression pin) | **16** |
| Relative energy error at p★ | ≤ 1e-10 |
| MAC | c = \|cA−cB\|/(rA+rB) ≥ 2, r = half·√3 |
| Leaf capacity | 8 |
| Random | 2×32 charges in balls of radius 0.5 at 0 and 20 (forces M2L) |
| Lattice | 2×(3³) alternating ±1 blocks centred 20 Å apart |
| Scaling smoke | N = 1024 vs 4N, p = 4, leaf = 16 |

## Units

Kernel is dimensionless q_i q_j / r. Conversion to eV (`units::coulomb`) is
intentionally **out of scope** here (chain link 21, CFMM / HF wiring).
