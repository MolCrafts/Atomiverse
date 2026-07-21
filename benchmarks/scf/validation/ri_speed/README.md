# ri_speed — glycine/def2-SVP RHF wall-clock: RI-JK vs exact ERI vs PySCF-DF

Speed evidence for hf-capability-gaps-12-df (density fitting). NOT part
of ctest; run manually via `run.sh`.

## Methodology

- **System**: gas-phase glycine (NH2-CH2-COOH, 10 atoms; geometry
  pinned identically in `bench_ri_speed.cpp` and the PySCF block of
  `run.sh`), RHF / def2-SVP (spherical), convergence |ΔE| ≤ 1e-9 Ha in
  all three columns (`rhf.tol = 1e-9` / `mf.conv_tol = 1e-9`).
- **Columns**: `ours-exact` = Atomiverse direct SCF (shell-block
  Schwarz-screened 4-index ERIs, `RI::None`); `ours-RI-JK` = the same
  SCF with `rhf.ri = RI::JK` over `AuxBasis::Def2JKFIT` (wall time
  INCLUDES the one-time DF workspace build); `PySCF-DF` =
  `scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')`,
  `mf.kernel()` wall time.
- **Repeats**: 3 per column, cold objects each repeat; per-run times
  printed, the table pins the median. Wall clock only
  (`std::chrono::steady_clock` / `time.perf_counter`).
- **Threads**: `OMP_NUM_THREADS=8`, `OMP_PROC_BIND=close`,
  `OMP_PLACES=cores` for all three columns (PySCF confirmed
  `lib.num_threads() == 8`).
- The engine binary is compiled by `run.sh` from the repo build tree,
  reusing the compile/link recipe of the `reg_h2o_ri_jk` regression
  target (no hardcoded toolchain paths; requires a build configured
  with `-DATV_BUILD_REGRESSIONS=ON`).

## Pinned result (run 2026-07-21)

Machine/env: x86_64 login node (2× AMD EPYC 9005-series, EL9), GCC
14.3.0 via `buildenv-gcc/2026.03-mpich`, OpenBLAS 0.3.32 (openmp,
lp64), `-O3 -march=native`, Release; PySCF 2.14.0 on
`~/work/.x86_64/bin/python`; 8 OpenMP threads.

| column | wall time [s] (median of 3) | E(RHF) [Ha] |
|---|---|---|
| ours-RI-JK (`rhf.ri = RI::JK`) | **1.617** | −282.615514331 |
| ours-exact (`RI::None`) | 2.516 | −282.615772070 |
| PySCF-DF (`density_fit()`) | 0.104 | −282.615514330 |

Per-run raw times: exact 2.508 / 2.519 / 2.516; RI-JK 1.617 / 1.618 /
1.616; PySCF-DF 0.121 / 0.102 / 0.104.

Observations pinned into ch19:

- RI-JK beats the exact-ERI path 1.56× at this (small) system size;
  the win grows with system size (the fitted build replaces the O(N⁴)
  quartet loop with O(N³)-shaped GEMM/loop work per iteration).
- Fit error: E(RI-JK) − E(exact) = +2.577e-4 Ha for 10 atoms
  (≈ 26 μHa/atom), the expected order for the matched def2 JKFIT set.
- Our RI-JK energy agrees with PySCF-DF to 1e-9 Ha (same spherical
  auxiliary span); the remaining ~15× wall-clock gap to PySCF-DF is
  general engine speed (SCF iteration count and integral kernels — see
  the GPW/SCF perf campaign), not an RI-algorithm difference.
- One-time DF workspace build (3-center integrals + metric Cholesky +
  B contraction): ~0.09 s of the 1.62 s, i.e. the RI win is in the
  per-iteration Fock build.
