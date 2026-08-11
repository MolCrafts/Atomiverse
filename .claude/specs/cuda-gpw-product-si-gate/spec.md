# cuda-gpw-product-si-gate

## Lever

| | |
| --- | --- |
| **What** | Si-class Γ RKS PADE **single-grid** energy dual-gate + fix H0/STVpp under Si nonlocal |
| **Files** | `src/cuda/gpw/gpw_solver.cu` `add_nonlocal_contracted` → KB via `add_nonlocal_kb_k@Γ` |
| **Metric** | Both SCF converge; \|E_cuda − E_cpu\| ≤ 5e-5 Ha on diamond Si bct ngrids=1 |
| **Kernel** | Host setup only (no new device kernel); multi-k KB already device-or-host shared |

## Why

H₂ full-product + water product-h2o gates are H / O+H GTH — Si DZVP-MOLOPT +
GTH-PBE-q4 nonlocal **ungated**. Tip `multigrid-si-energy-path` GH200
**1144059**/**1144870**: first-iter E −36.35 vs CPU −14.97 even with full host
collint KS → primary residual **H0/STVpp** under Si nonlocal
(`add_nonlocal_contracted` ket-only images vs CPU `add_nonlocal_kb`).

## Twin

Same-settings CPU diamond Si bct 4-atom, DZVP-MOLOPT-GTH / GTH-PBE-q4, Γ,
CUTOFF 40 Ry, ngrids=1, n_occ=8, PADE, DIIS. Oracle: **CPU twin**.

## Acceptance

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| Both SCF converged | required |
| GH200 log | `GPU_JOB_OK` / `PRODUCT_SI_GATE_PASS` |
| Mid-path scientific DH | zero new mid-SCF matrix DH (setup-only change) |

**perf_exempt** — small-cell correctness.

## Non-goals

MG / multi-k / F/σ / continuous collocate PBC (`product-collocate-pbc`).
