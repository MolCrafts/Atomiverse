# Reference — `mpi-gpw-scaling-4rank`

## Method

Closed-shell **GPW RKS** with **Monkhorst–Pack k-point sampling** and
**MPI k-point round-robin** over `MPI_COMM_WORLD` (mirrors plane-wave
`cpu::pw` layout: `ik = rank; ik < n_k; ik += nprocs`). Each rank
collocates / diagonalizes only its local k; real-space density,
kinetic/nonlocal energy scalars, DIIS \(B\)-matrix partials, and residual
norms are Allreduced so every rank holds identical `e_total`.

Public API: concrete `cpu::gpw::GpwDft` members (`k_mesh`, `cutoff_ry`, …);
`GpwResult::{n_kpts, n_kpts_local, e_total, …}`. No public `*Config`, no
factories.

## System (lightweight twin of `gpw-kpoints-si`)

| | |
| --- | --- |
| Lattice | Si diamond bct 4-atom, a = 5.43 Å |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| XC | PBE |
| Density cutoff | **80 Ry** (e2e self-launch budget) |
| k mesh | MONKHORST-PACK 2×2×2, TR → **4 k** |
| SCF | DIIS, tol 1e-7 Ha |

## What is checked

1. Serial SCF converges; `n_kpts == 4`, `n_kpts_local == n_kpts`.
2. Parent self-launches `mpiexec -n 4 <binary> --mpi-worker`.
3. Worker: all ranks agree on energy (\(\le 10^{-12}\) Ha spread); local
   k counts sum to `n_kpts`.
4. Rank-count invariance \(|E_{n=1}-E_{n=4}| \le 10^{-8}\) Ha.
5. Wall ratio \(T_1/T_4\) recorded; **perf_exempt**.

## Measured (first green, 2026-07-25, OMP=1)

| Quantity | Value |
| --- | --- |
| E (serial n=1, 80 Ry MP 2×2×2 TR) | −15.8024163941 Ha (6 iters) |
| E (mpiexec n=4) | −15.8024163941 Ha |
| \|E1 − E4\| | **0** (machine-identical) |
| T1 / T4_worker | ≈ 0.995 (setup still replicated — no speedup expected) |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | k-RR foothold only (setup matrices still replicated; no real-space domain decomp); no CP2K multi-rank twin required this wave |
| r = T_atv/T_cp2k | n/a (exempt) |
| wall ratio T1/T4 | ≈ 1.0 (informational; k-RR alone does not beat serial when setup is O(nk) replicated) |

## Honest scope

- **Shipped:** multi-k SCF k-point RR + density/energy/DIIS Allreduce;
  serial bit-identical when nprocs=1 / MPI not initialized.
- **Not shipped:** Γ-only MPI work sharing; grid/AO domain decomposition;
  k-UKS; k-point forces under MPI.
