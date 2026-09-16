# Geometry Optimization and Reaction Paths

Legend and caveats: [Roadmap index](index.md).

The minimizer preset (`cpu::minimizer::Minimize`, `cuda::minimizer::Minimize`)
hosts the optimizer as a hot-swappable Component, so any force-providing
evaluator — LJ, bonded, SCF, GPW — can be relaxed by any minimizer. FIRE 1.0
ships as `cpu::minimizer::FIRE`. **Variable-cell** ships isotropic, joint
isotropic atom+cell, **anisotropic**, and **triclinic** stress relax
(`IsotropicCellRelax`, `JointIsotropicRelax`, `AnisotropicCellRelax`,
`TriclinicCellRelax`). The convergence rule is a `ForceConverged`
(fixed cell), `PressureConverged` (isotropic) or `StressConverged` (per-axis,
maskable) criterion — a different stopping rule costs a criterion, not an
optimizer.

**Paths:** CI-NEB, dimer, **string**, **IRC** ship. Still absent:
eigenvector-following TS.

## 1. Geometry Optimization

| Method                                     | CPU | CUDA | Kokkos |
| ------------------------------------------ | :-: | :--: | :---: |
| Steepest descent                           |  ○  |   ✓  |   ○  |
| Polak–Ribière conjugate gradient           |  ✓  |   ✓  |   ○  |
| Nonlinear conjugate gradient               |  ✓  |   ✓  |   ○  |
| L-BFGS                                     |  ✓  |   ✓  |   ○  |
| Trust-region optimization                  |  ○  |   ○  |   ○  |
| Rational-function optimization             |  ○  |   ○  |   ○  |
| Partitioned rational-function optimization |  ○  |   ○  |   ○  |
| Augmented-Hessian optimization             |  ○  |   ○  |   ○  |
| FIRE minimization                          |  ✓  |   ○  |   ○  |
| Cartesian-coordinate optimization          |  ✓  |   ✓  |   ○  |
| Internal-coordinate optimization           |  ○  |   ○  |   ○  |
| Redundant-internal-coordinate optimization |  ○  |   ○  |   ○  |
| Constrained geometry optimization          |  ○  |   ○  |   ○  |
| Variable-cell optimization                 |  ✓  |   ○  |   ○  |
| Joint atomic and cell optimization         |  ✓  |   ○  |   ○  |

Notes:

- **Steepest descent is CUDA-only** as a distinct component (`cuda/minimize/sd.cuh`).
  On CPU, SD exists only as the degenerate β = 0 limit of CG — there is no SD
  component to select.
- The CG flavour is **Polak–Ribière⁺** (restart by β clamp) on both backends;
  Fletcher–Reeves is not implemented. It is a nonlinear CG, hence both rows ✓.
- Line search on CPU offers fixed-step, Armijo backtracking, and **strong Wolfe**
  (Nocedal & Wright bracket + zoom). The CUDA L-BFGS uses Armijo backtracking.
- Convergence on CUDA checks five criteria (max/RMS force, max/RMS displacement,
  ΔE); the CUDA SD component deliberately runs its full budget without one.
- **Variable-cell:** isotropic / anisotropic / triclinic optimizers + joint
  atom+cell isotropic. Classical virial and **PW electronic stress**
  (Nielsen–Martin analytic `pw_stress`, mechanical / CP2K sign) drive cell
  steps; **GPW stress** is analytic fixed-\(P\)+Pulay (`gpw_stress`; multi-SCF
  FD oracle `gpw_stress_fd`; `reg_gpw-analytic-stress` /
  `reg_gpw-stress-water`). **GPW CELL_OPT is productized**:
  `AnisotropicCellRelax` re-targets the GPW evaluator with a stress axes mask
  + `StressConverged` (`reg_gpw-variable-cell-ice` vs CP2K CELL_OPT,
  2026-07-25). **PW CELL_OPT is productized**: `IsotropicCellRelax` +
  `StressConverged` on diamond Si NC-UPF (`reg_pw-variable-cell-si`,
  2026-07-25). `kStress` is the **extensive** Voigt virial [eV] for every
  producer (`core/frame_schema.h`): PW/GPW multiply their intensive σ by the
  cell volume at the Frame write, and every consumer — the four relaxers,
  `PressureConverged` and `StressConverged` — divides by \(V\).
  No internal coordinates. **PR NPT
  dynamics** lives under [Molecular Dynamics](molecular-dynamics.md)
  (`ParrinelloRahmanNPT`), not here.

## 2. Transition-State and Reaction-Path Methods

| Method                                        | CPU | CUDA | Kokkos |
| --------------------------------------------- | :-: | :--: | :---: |
| Eigenvector-following transition-state search |  ○  |   ○  |   ○  |
| Partitioned-RFO transition-state search       |  ○  |   ○  |   ○  |
| Dimer method                                  |  ✓  |   ○  |   ○  |
| Climbing-image nudged elastic band            |  ✓  |   ○  |   ○  |
| Adaptive nudged elastic band                  |  ○  |   ○  |   ○  |
| Growing string method                         |  ✓  |   ○  |   ○  |
| Freezing string method                        |  ○  |   ○  |   ○  |
| Intrinsic reaction coordinate                 |  ✓  |   ○  |   ○  |
| Minimum-energy path optimization              |  ✓  |   ○  |   ○  |
| Minimum-energy crossing point search          |  ○  |   ○  |   ○  |
| Conical-intersection optimization             |  ○  |   ○  |   ○  |
| Basin hopping                                 |  ○  |   ○  |   ○  |
| Minima hopping                                |  ○  |   ○  |   ○  |
