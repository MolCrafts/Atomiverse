# Geometry Optimization and Reaction Paths

Tutorial: [Geometry optimization](../tutorial/ch12-geometry-optimization.md).

`cpu::minimizer::Minimize` / `cuda::minimizer::Minimize` host any
force-providing evaluator. Convergence is a Criterion
(`ForceConverged` / `PressureConverged` / `StressConverged`), not an optimizer
flag. Legend: [Roadmap index](index.md).

## 1. Minimizers

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| L-BFGS | ✓ | ✓ | CPU: a failed line search resets the history and retries steepest descent, then stops unconverged (`lbfgs_line_search_failed`) instead of raising; CG restarts the same way |
| Polak–Ribière⁺ CG | ✓ | ✓ | Nonlinear; Fletcher–Reeves not shipped |
| FIRE | ✓ | ○ | |
| Steepest descent | ○ | ✓ | CPU: only as CG β = 0; no SD component |
| Cartesian coordinates | ✓ | ✓ | |
| Internal / redundant internals | ○ | ○ | |
| Variable-cell (iso / aniso) | ✓ | ○ | + joint atom+cell; cell geometry must be supported by the evaluator |
| Variable-cell (triclinic) | ◐ | ○ | No external pin; self-pinned LJ shear case removed |
| Constrained opt | ○ | ○ | |

CPU PME/PPPM use `FFTW_ESTIMATE` plans, so a minimization or trajectory is
bit-reproducible across processes for a given thread count (a timed
`FFTW_MEASURE` plan could change the rounding between runs).
Cell steps consume extensive `kStress` [eV]; consumers divide by \(V\).
GPW/PW CELL_OPT productized. PR-NPT dynamics lives under
[MD](molecular-dynamics.md), not here.
CPU `CellList` evaluators require orthorhombic cells and reject shear;
triclinic cell updates require an evaluator with triclinic support.

## 2. Transition states and paths

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| CI-NEB | ✓ | ○ | |
| Dimer | ✓ | ○ | |
| Growing string | ✓ | ○ | |
| IRC | ✓ | ○ | |
| Eigenvector-following TS | ○ | ○ | |
| MECP / conical intersection | ○ | ○ | |
| Basin / minima hopping | ○ | ○ | |
