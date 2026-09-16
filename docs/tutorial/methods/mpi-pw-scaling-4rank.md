# MPI plane-wave k-mesh scaling — 4 ranks

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `mpi-pw-scaling-4rank` |
| Regression | `regressions/pw/mpi-pw-scaling-4rank/` |
| Public API | `cpu::pw::PW` (members) + `PWResult::{n_kpts,n_kpts_local}` |
| Roadmap | [dft.md](../../roadmap/dft.md) § plane-wave DFT (k-point RR MPI)|
| Dep | `pw-lda-si` ✓ (Si ONCV LDA twin); sibling gate `mpi-gpw-scaling-4rank` |

## Summary

When Atomiverse is built with MPI, plane-wave SCF (`cpu::pw::PW`) distributes
Monkhorst–Pack **k-points round-robin** over `MPI_COMM_WORLD`. Each rank owns
k-points \(\{r,\, r+n_\mathrm{procs},\, \ldots\}\), diagonalizes those
subspaces with Davidson, then **Allreduces** the real-space density and
band-energy sum every SCF iteration so every rank holds the identical mixed
density and takes identical convergence decisions. The Davidson initial guess
is seeded by subspace index (not rank), so the SCF trajectory is **rank-count
invariant** by construction.

This slug is the **industrial gate** for that claim: the same Si diamond
ONCV-LDA twin as `pw-lda-si` at **np = 1** and **np = 4**, plus a weak wall
ratio floor. Absolute energy is pinned to the PseudoDojo ABINIT-class
incomplete-basis band; performance vs CP2K is **`perf_exempt`** when no
CP2K binary is available on the host.

## Observables

| Symbol | Member | Meaning |
| ------ | ------ | ------- |
| \(n_k\) | `PWResult::n_kpts` | Total MP k-points of the run (`k_mesh`³ set) |
| \(n_k^\mathrm{local}\) | `PWResult::n_kpts_local` | k-points owned by **this** rank under RR |
| \(E\) | `PWResult::energy` | Total energy [Ry] |
| \(T_\mathrm{run}\) | regression wall | `MPI_Wtime` around `run()` (max over ranks) |

Serial / MPI-inactive: \(n_k^\mathrm{local} = n_k\). Partition identity:

\[
\sum_r n_k^\mathrm{local}(r) = n_k,
\qquad
\max_r n_k^\mathrm{local} - \min_r n_k^\mathrm{local} \le 1.
\]

For this twin (\(k_\mathrm{mesh}=2\) → 8 k-points) at 4 ranks each rank owns
exactly 2.

## Usage

```cpp
cpu::pw::PW pw;
pw.upf_path = "tests/data/pw/Si_lda_nc.upf";
pw.ecut_ry  = 15.0;
pw.k_mesh   = 2;       // 8 k-points
pw.tol      = 1e-6;
pw.max_iter = 50;

pw.build(frame);
const PWResult& r = pw.run(frame);   // under mpiexec -n N

// Per-rank distribution (gate-readable):
// r.n_kpts        — global count
// r.n_kpts_local  — this rank's owned count
```

No public `*Config` bag and no factory: members on `PW`, Frame in / result out.

## Validation (regression `reg_mpi-pw-scaling-4rank`)

Twin: diamond Si 2-atom fcc, \(a=5.43\) Å, ONCV LDA UPF
(`tests/data/pw/Si_lda_nc.upf`), ecut 15 Ry, \(2\times2\times2\) MP,
tol \(10^{-6}\), `max_iter` 50. Equal `OMP_NUM_THREADS` per rank in both
launches. The binary self-launches via the build-configured `mpiexec`
(`ATV_MPIEXEC`); not part of default ctest — run
`cmake --build build --target run_reg_mpi-pw-scaling-4rank`.

| Check | Bound |
| ----- | ----- |
| Energy (np=1 and np=4) | \([-16.916316,\, -16.916316 + 1.50]\) Ry (PseudoDojo ABINIT \(E_\mathrm{min}\)) |
| Rank-count invariance | \(\lvert E(1)-E(4)\rvert \le 10^{-8}\) Ry; identical SCF iters |
| 4-way split | `nk_local_min == nk_local_max == 2` at np=4 |
| Speedup | \(T(1)/T(4) \ge 1.2\) (weak floor; grid work is still replicated) |
| Budget | total wall ≤ 60 s |

Unit gate (`tests/mpi/test_pw_kpoint.cpp`, ctest at n = 1/2/4):
`Allreduce SUM(n_kpts_local) == n_kpts` and `max − min ≤ 1` after one SCF
pass (`max_iter = 1`).

## Boundaries

- **k-point RR only** — no G-vector / band / domain decomposition in this slug.
- CPU plane-wave path; CUDA PW MPI is out of scope.
- GPW MPI scaling is the sibling slug `mpi-gpw-scaling-4rank`.
- No live CP2K/ABINIT/QE at ctest or regression time (offline pins only).
- Grid and density work remain replicated on every rank at this problem size;
  only the per-k Davidson diagonalization parallelizes — the speedup floor
  is therefore deliberately weak.
