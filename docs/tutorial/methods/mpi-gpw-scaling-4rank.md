# MPI GPW k-point scaling — 4-rank industrial gate

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `mpi-gpw-scaling-4rank` |
| Regression | `regressions/gpw/mpi-gpw-scaling-4rank/` |
| Public API | `cpu::gpw::GpwDft` multi-k (`k_mesh` / `k_points`); `GpwResult::{n_kpts, n_kpts_local, e_total}` |
| Unit | `tests/mpi/test_gpw_kpoint.cpp` (CTest `mpiexec` n=1/2/4) |
| Dual gate | Rank-count energy invariance; speed **`perf_exempt`** (k-RR foothold only) |

## Summary

**MPI k-point round-robin** for the closed-shell GPW multi-k RKS SCF, matching
the plane-wave pattern already shipping in `cpu::pw`:

\[
i_k \in \{\,r,\; r+P,\; r+2P,\; \ldots\,\},
\quad r = \mathrm{rank},\; P = n_{\mathrm{procs}}.
\]

Each rank collocates only its local Bloch density contribution, builds and
diagonalizes \(H_{k}\) for those points, and accumulates weighted kinetic /
nonlocal traces. After the local work:

- `MPI_Allreduce(SUM)` on \(\rho(\mathbf{r})\) (full real-space grid),
- `MPI_Allreduce(SUM)` on \(E_{\mathrm{kin}}\), \(E_{\mathrm{nl}}\),
- `MPI_Allreduce(MAX)` on the DIIS residual norm,
- `MPI_Allreduce(SUM)` on the Pulay \(B_{ij}\) block inside k-DIIS,

so every rank obtains the **same** total energy and the same SCF trajectory.
When MPI is not initialized or `nprocs == 1`, the distributed branches are
inactive and the path is **bit-identical** to the pre-MPI serial k-path.

**Honest partial:** setup of \(S_k\), \(T_k\), \(V_{\mathrm{nl}}^k\), and the
pre-summed complex basis-on-grid remains **replicated** on every rank. This
wave does **not** distribute the real-space density grid or FFT (full GPW
domain decomposition is a later industrial step). Fair wall vs CP2K is
**`perf_exempt`**.

## Usage

```cpp
cpu::gpw::GpwDft gpw;
gpw.cell[0] = gpw.cell[1] = a / std::sqrt(2.0); // bct Si
gpw.cell[2] = a;
gpw.cutoff_ry = 150.0;
gpw.n_occ = 8;
gpw.k_mesh[0] = gpw.k_mesh[1] = gpw.k_mesh[2] = 2;
gpw.k_time_reversal = true;
gpw.xc = cpu::gpw::XcKind::PBE;
gpw.build(atoms);
const auto& r = gpw.run(atoms);
// r.e_total identical on every rank under mpiexec
// r.n_kpts == 4 (TR-folded 2×2×2); r.n_kpts_local = ceil partition
```

Launch (library must be built with `-DATV_USE_MPI=ON`):

```bash
mpiexec -n 4 ./your_gpw_app
# or the regression self-launcher:
./reg_mpi-gpw-scaling-4rank
```

## Pin

- **Unit:** `test_mpi_gpw_kpoint` — all ranks agree on `e_total` to \(10^{-12}\)
  Ha; `sum(n_kpts_local) == n_kpts`.
- **E2E:** `reg_mpi-gpw-scaling-4rank` — parent serial energy vs self-launched
  `mpiexec -n 4` worker; \(|E_1 - E_4| \le 10^{-8}\) Ha; wall ratio recorded.

See `regressions/gpw/mpi-gpw-scaling-4rank/REFERENCE.md`.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | k-point RR only; setup still replicated; no CP2K multi-rank twin this wave |

## Related

- Serial k-point correctness: `gpw-kpoints-si`
- PW MPI k-RR twin: `tests/mpi/test_pw_kpoint.cpp`
- Spec: `.claude/specs/mpi-gpw-scaling-4rank/`
