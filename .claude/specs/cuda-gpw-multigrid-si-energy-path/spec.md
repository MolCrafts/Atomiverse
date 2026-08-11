# cuda-gpw-multigrid-si-energy-path

## Goal

Green **Si-class GTH Γ RKS PADE multigrid** (`ngrids>1` + `rel_cutoff>0`) product
**energy path** on public `cuda::gpw::GpwDft` after parents:

- `cuda-gpw-product-si-gate` — Si H0/KB Vnl + SG energy green
- `cuda-gpw-product-collocate-pbc` — sphere-window PBC collocate/integrate
  (shared by all MG levels)

H₂ `multigrid-run` foothold alone left production Si nonlocal × multi-level
ungated / historically non-convergent.

## Lever (named before code)

| | |
| --- | --- |
| **Files** | `src/cuda/gpw/gpw_solver.cu` (`build_energy_device` MG branch), `src/cuda/gpw/internal/{product_collocate,product_integrate,ks_chain,multigrid_transfer}.cu` |
| **Function** | mid-SCF MG: per-level `pack_weights` → `KsMultiGridChainDevice::run` (PBC collocate + prolong + fine KS + restrict + integrate) → `scatter_acc` / `scatter_acc_add` → device `H=H0+V` / DIIS |
| **Metric** | both SCF converge; \|E_cuda − E_cpu\| ≤ **5×10⁻⁵ Ha** on diamond Si Γ ngrids=4 |
| **Not** | new dual-gate around mid-path DH; MG F/σ; multi-k MG; soft-tol |

Parents already closed: PBC collocate charge under-count; Si KB H0 residual.
This slug **re-verifies continuous device MG KS** under Si production AO/grid
and lands the dual-gate. Path surgery only if dual-gate fails.

## Acceptance

1. Spec-lite present (this file + acceptance.md).
2. Offline dual-gate reg `reg_cuda-gpw-multigrid-si-energy-path` vs same-settings
   CPU twin (diamond Si bct 4-atom, DZVP-MOLOPT-GTH / GTH-PBE-q4, Γ, CUTOFF 40,
   ngrids=4, rel_cutoff=20, PADE, DIIS).
3. GH200: both converge + \|dE\| ≤ 5e-5 Ha (no soft-tol / self-golden).
4. Mid-path scientific matrix DH grep on own diff = **zero** (scalar residual OK).
5. Skills named if any `.cu` kernel edit (else compose-only).
6. **perf_exempt** — correctness path-fix, not fair-wall.

## Non-goals

Water MG, multi-k MG, MG F/σ, Si MG×PBE, device-payload / STVnl / residency
rows already named elsewhere. Child `multigrid-si-gate` owns formal production
polish only after this parent greens.
