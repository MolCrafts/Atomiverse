# CFMM vs direct J-build scaling (alkane series)

## Purpose

Identify the system-size **crossover** beyond which CFMM-J is strictly
faster than Schwarz-direct J for closed-shell RHF / STO-3G alkanes
`C_n H_{2n+2}`. Numbers feed the ch19 tutorial table.

## Methodology

- Series: n = 4, 6, 8, 10, 12, 16 (and optionally 20) all-trans alkanes.
- Basis: STO-3G. Coulomb: `Coulomb::Direct` vs `Coulomb::CFMM`.
- CFMM knobs: `cfmm_order = 10`, `cfmm_threshold = 1e-9`.
- Metric: wall-clock of **one** J-build (or a fixed small number of SCF
  iterations with the J phase timed), median of 3 runs, single node.
- Accuracy guard: each CFMM run must satisfy
  `|E_CFMM − E_Direct| ≤ 1e-7 Ha` before its timing is accepted.
- **Not** registered in ctest.

## Machine env (fill when run)

| Field | Value |
|-------|-------|
| Host | _TBD_ |
| CPU | _TBD_ |
| Threads (`OMP_NUM_THREADS`) | _TBD_ |
| Build | Release, `-DATV_BUILD_HF=ON -DATV_BUILD_CUDA=OFF` |
| Date | _TBD_ |

## Results table (fill when run)

| n | N_atoms | N_AO | t_Direct [s] | t_CFMM [s] | speedup |
|---|---------|------|--------------|------------|---------|
| 4 | 14 | ~26 | — | — | — |
| 6 | 20 | ~38 | — | — | — |
| 8 | 26 | ~50 | — | — | — |
| 10 | 32 | ~62 | — | — | — |
| 12 | 38 | ~74 | — | — | — |
| 16 | 50 | ~98 | — | — | — |

Crossover size: _TBD_ (first n where t_CFMM < t_Direct).

## How to run

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  export OMP_NUM_THREADS=8
  ./run_cfmm_scaling.sh
'
```

Driver script `run_cfmm_scaling.sh` times `reg_c10h22_cfmm_j`-style RHF
loops over the alkane series (or a dedicated bench binary when added).
