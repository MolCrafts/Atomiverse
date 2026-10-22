# GPW DFT+U — projector divergence vs CP2K, and the variational scatter

## Two different Mulliken projectors

| | Occupation on the correlated (atom, ℓ) block | Geometry dependence |
|---|---|---|
| **ATV** (`cpu/gpw/internal/dft_plus_u.h`) | symmetrized **gross** product ½(PS+SP)\|_block (BZ-averaged on k) | depends on the full S ⇒ +U has an occupation-matrix **force/stress** term |
| **CP2K `PLUS_U_METHOD MULLIKEN`** (`src/dft_plus_u.F`, `mulliken`) | on-site **net** Hadamard block q_ab = P_ab·S_ab | on-site S is position-independent ⇒ **no** +U force term |
| CP2K `MULLIKEN_CHARGES` | gross AO charge vector q_i = Σ_j P_ij S_ij | geometry-dependent; CP2K adds the W-correction `w −= ½(dEdq_i+dEdq_j)P_ij` |

Consequences:

- +U **energies** agree only loosely (e.g. NiO molecule twin: e_dftu ATV ≈ 0.072
  vs CP2K 0.0495 Ha; asserted only through E_tot at 5e-2).
- +U **force contributions differ by model, not bug**: stretched NiO ΔF_z is
  +9.0e-2 Ha/Bohr (ATV, confirmed against ATV's own FD to 3.7e-4) vs
  −1.1e-3 (CP2K). No honest cross-projector force tolerance exists.
- A future loop row may align ATV to CP2K's net-population MULLIKEN (or add it
  as a second method). That rewrites **every** +U fixed point and re-pins the
  +U regression family — schedule deliberately, not as a drive-by.

## Variational scatter

For ATV's occupation the exact Fock contribution is ∂E_U/∂P = **½(V̂S+SV̂)**.
A bare-V̂ scatter leaves the SCF fixed point non-stationary for the reported
E_U[n(P,S)]: analytic-vs-FD forces then disagree at O(1) Ha/Bohr on the H₂ toy
and +U stress dual gates carry a ~1e-3 Ha/Bohr³ residual (3e-9 when correct).

## Toy-regime warning

The H₂ DZV s-shell toy at U−J = 4 eV sits in the strong-overlap regime:
gross-Mulliken occupation eigenvalues leave [0,1] (CP2K warns "unphysical
Mulliken charges"), the SCF is multi-branch and slow (90–250 DIIS iters), and
displaced/strained FD oracles can hop branches. Force-level dual gates on
this toy must use U−J ≈ 2 eV (single branch); symmetric geometries behave
better than asymmetric ones (the asymmetric `cheap_h2` k+U stress toy
oscillated until moved to the symmetric bond).

NiO +U DIIS convergence is slow near the fixed point: tol 1e-8 needs ~480
iters — offline-FD only; in-budget regressions gate forces on the H₂ multi-k
full-6N dual gate instead.
