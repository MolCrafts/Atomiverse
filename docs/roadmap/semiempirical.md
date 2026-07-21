# Semiempirical and Tight-Binding Methods

Legend and caveats: [Roadmap index](index.md).

Atomiverse ships one semiempirical method: **AM1**, on the NDDO framework
(`cpu::semiempirical::NDDOSolver` + `AM1Evaluator` of kind `cpu::Semiempirical`).
It computes energies, Mulliken charges, the density matrix, and **analytic
nuclear forces** (SCF Hellmann–Feynman / fixed-density integral derivatives).
`AM1Evaluator::with_gradient(true)` (default) writes forces for Evaluate / MD /
Minimize; `with_gradient(false)` is energy-only. Launch unwraps an isolated
fragment for molecular AM1 under PBC wrap (same pattern as Gaussian SCF).
Regressions: `am1_water_force`, `am1_water_minimize`, `am1_water_bomd`.
Tutorial: [Chapter 17](../tutorial/ch17-semiempirical-am1.md) +
`examples/15_am1.cpp`. Heat of formation is MOPAC-style on `AM1Result`.
Force gradients use fixed-density central differences with translational
invariance (last atom \(F_N=-\sum_{i<N} F_i\)); closed-form multipole
derivatives remain a performance follow-up.
AM1-BCC charge assignment still uses the energy/charge path (see
[Multiscale and Properties](multiscale.md)).

There is no DFTB and no extended tight binding (xTB/GFN-n) anywhere in the tree.
Those are **separate method families** (own Solver + Evaluator + kind later) —
never a new Driver, and not under `NDDOSolver`.

## 1. Semiempirical Quantum Methods

| Method                          | CPU | CUDA |
| ------------------------------- | :-: | :--: |
| AM1                             |  ✓  |   ○  |
| RM1                             |  ✓  |   ○  |
| PM6                             |  ○  |   ○  |
| PM7                             |  ○  |   ○  |
| OM3                             |  ○  |   ○  |
| Semiempirical analytic gradient |  ✓  |   ○  |
| Semiempirical analytic Hessian  |  ○  |   ○  |
| AM1 heat of formation (MOPAC-style ΔHf) |  ✓  |   ○  |

The NDDO core (overlap, one-electron core Hamiltonian, two-electron integrals,
Fock build, density, SCF, forces) is parameterization-agnostic in structure.
**AM1** and **RM1** (H/C/N/O/P/S; Rocha et al. 2006 / MOPAC tables) are
transcribed. PM6/PM7 remain tables away. Analytic Hessian remains
finite-difference via `cpu::vib::Hessian` over SE forces.

## 2. Density-Functional Tight Binding

| Method                    | CPU | CUDA |
| ------------------------- | :-: | :--: |
| SCC-DFTB / DFTB2          |  ○  |   ○  |
| DFTB3                     |  ○  |   ○  |
| Spin-polarized DFTB       |  ○  |   ○  |
| Noncollinear-spin DFTB    |  ○  |   ○  |
| DFTB+U                    |  ○  |   ○  |
| Dispersion-corrected DFTB |  ○  |   ○  |
| Spin-orbit DFTB           |  ○  |   ○  |
| Periodic DFTB             |  ○  |   ○  |
| DFTB analytic gradient    |  ○  |   ○  |
| DFTB analytic stress      |  ○  |   ○  |
| TD-DFTB                   |  ○  |   ○  |

## 3. Extended Tight-Binding Methods

| Method                    | CPU | CUDA |
| ------------------------- | :-: | :--: |
| GFN0-xTB                  |  ○  |   ○  |
| GFN2-xTB                  |  ○  |   ○  |
| Spin-polarized GFN-xTB    |  ○  |   ○  |
| IPEA-xTB                  |  ○  |   ○  |
| Periodic GFN-xTB          |  ○  |   ○  |
| GFN-xTB analytic gradient |  ○  |   ○  |
| GFN-xTB analytic Hessian  |  ○  |   ○  |
| GFN-FF                    |  ○  |   ○  |
| sTDA-xTB                  |  ○  |   ○  |
