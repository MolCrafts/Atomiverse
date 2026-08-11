# Acceptance — cuda-gpw-multigrid-si-energy-path

| Check | Criterion |
| --- | --- |
| Twin | same-settings CPU `GpwDft` diamond Si Γ PADE ngrids=4 rel_cutoff=20 |
| Energy | \|E_cuda − E_cpu\| ≤ 5e-5 Ha |
| Converge | both SCF `converged=true` |
| Residency | no mid-SCF scientific matrix D2H/H2D in MG energy body (boundary payload OK) |
| Runtime | GH200 log required; login has no toolkit |
| Perf | **perf_exempt** (path correctness, not CP2K fair-wall) |
| Scope | energy only — not F/σ / multi-k / PBE |

When green: `acceptance_met=true`, delete this spec dir, mark queue row [x].
