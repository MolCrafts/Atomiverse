# C10H22 / STO-3G CFMM-J — reference

## What is under test

Continuous fast multipole method (CFMM) for the HF Coulomb matrix J on
n-decane / STO-3G, with multipole order `cfmm_order = 10` and extent
threshold `cfmm_threshold = 1e-9`. Exchange (K) stays on the exact
Schwarz path.

## Oracle

**In-tree Schwarz-screened direct J** (the vacuum path already pinned by
PySCF on earlier chain links). **PySCF does not ship an FMM/CFMM** for
molecular Coulomb — its default is direct SCF + Schwarz (optional DF).
There is therefore no external FMM numerical oracle; correctness is
defined against our own Direct path.

## Citations

- White, Johnson, Gill, Head-Gordon, *Chem. Phys. Lett.* **230**, 8
  (1994), DOI [10.1016/0009-2614(94)01128-1](https://doi.org/10.1016/0009-2614(94)01128-1)
  — Gaussian extent classification and continuous FMM.
- Greengard & Rokhlin, *J. Comput. Phys.* **73**, 325 (1987) — FMM
  skeleton (tree operators from chain link 20).

## Gates

| Gate | Quantity | Tolerance |
|------|----------|-----------|
| Energy | \|E_CFMM − E_Direct\| | ≤ 1e-7 Ha |
| J matrix (unit test) | max_ij \|J_CFMM − J_direct\| on H2O/STO-3G | ≤ 1e-8 Ha |

The elementwise J hard gate lives in `tests/cpu/scf/test_cfmm_j.cpp`
(out-of-library regressions cannot reach the private J matrix). This
regression pins the **converged total energy** dual gate on C10H22.

## Geometry

All-trans n-decane skeleton (C–C 1.54 Å, C–H 1.09 Å), 32 atoms, cubic
box 80 Å. Coordinates generated in `reg_c10h22_cfmm_j.cpp`.
