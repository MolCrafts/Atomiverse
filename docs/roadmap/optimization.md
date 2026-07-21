# Geometry Optimization and Reaction Paths

Legend and caveats: [Roadmap index](index.md).

The minimizer preset (`cpu::minimizer::Minimize`, `cuda::minimizer::Minimize`)
hosts the optimizer as a hot-swappable Component, so any force-providing
evaluator — LJ, bonded, SCF, GPW — can be relaxed by any minimizer. What is
missing is breadth of algorithm (no FIRE, no quasi-Newton beyond L-BFGS) and,
more consequentially, **any notion of relaxing the cell**. The convergence rule
itself is no longer part of any of this: it is a `cpu::ForceConverged` criterion,
and a different stopping criterion costs a criterion component, not an optimizer.

**Climbing-image NEB runs** (`cpu::PathDriver` + `cpu::path::NebCoupling` +
`cpu::path::BandLBFGS`) — a coupled chain of images terminating when *every*
image has converged (the `AllReplicasConverged` criterion), and the engine's only
multi-replica driver. The replicas are generated from one Frame by a
`ReplicaExpansion` strategy, so `run()` still takes one Frame and returns one
(for CI-NEB, the saddle configuration).

What is still absent is every *other* transition-state method: no dimer, no
eigenvector-following, no string method, and no IRC.

## 1. Geometry Optimization

| Method                                     | CPU | CUDA |
| ------------------------------------------ | :-: | :--: |
| Steepest descent                           |  ○  |   ✓  |
| Polak–Ribière conjugate gradient           |  ✓  |   ✓  |
| Nonlinear conjugate gradient               |  ✓  |   ✓  |
| L-BFGS                                     |  ✓  |   ✓  |
| Trust-region optimization                  |  ○  |   ○  |
| Rational-function optimization             |  ○  |   ○  |
| Partitioned rational-function optimization |  ○  |   ○  |
| Augmented-Hessian optimization             |  ○  |   ○  |
| FIRE minimization                          |  ○  |   ○  |
| Cartesian-coordinate optimization          |  ✓  |   ✓  |
| Internal-coordinate optimization           |  ○  |   ○  |
| Redundant-internal-coordinate optimization |  ○  |   ○  |
| Constrained geometry optimization          |  ○  |   ○  |
| Variable-cell optimization                 |  ○  |   ○  |
| Joint atomic and cell optimization         |  ○  |   ○  |

Notes:

- **Steepest descent is CUDA-only** as a distinct component (`cuda/minimizer/sd.cuh`).
  On CPU, SD exists only as the degenerate β = 0 limit of CG — there is no SD
  component to select.
- The CG flavour is **Polak–Ribière⁺** (restart by β clamp) on both backends;
  Fletcher–Reeves is not implemented. It is a nonlinear CG, hence both rows ✓.
- Line search on CPU offers fixed-step, Armijo backtracking, and **strong Wolfe**
  (Nocedal & Wright bracket + zoom). The CUDA L-BFGS uses Armijo backtracking.
- Convergence on CUDA checks five criteria (max/RMS force, max/RMS displacement,
  ΔE); the CUDA SD component deliberately runs its full budget without one.
- **Every minimizer moves atoms only.** No cell degrees of freedom, no
  constraints, no internal coordinates. Combined with the missing periodic stress
  tensor (see [Periodic DFT](dft.md#11-periodic-response-and-analysis)),
  variable-cell relaxation is doubly
  blocked: no stress to relax against, and no optimizer that would accept it.

## 2. Transition-State and Reaction-Path Methods

| Method                                        | CPU | CUDA |
| --------------------------------------------- | :-: | :--: |
| Eigenvector-following transition-state search |  ○  |   ○  |
| Partitioned-RFO transition-state search       |  ○  |   ○  |
| Dimer method                                  |  ○  |   ○  |
| Climbing-image nudged elastic band            |  ✓  |   ○  |
| Adaptive nudged elastic band                  |  ○  |   ○  |
| Growing string method                         |  ○  |   ○  |
| Freezing string method                        |  ○  |   ○  |
| Intrinsic reaction coordinate                 |  ○  |   ○  |
| Minimum-energy path optimization              |  ✓  |   ○  |
| Minimum-energy crossing point search          |  ○  |   ○  |
| Conical-intersection optimization             |  ○  |   ○  |
| Basin hopping                                 |  ○  |   ○  |
| Minima hopping                                |  ○  |   ○  |
