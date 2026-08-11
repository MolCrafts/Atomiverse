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
| **Files** | `src/cuda/gpw/gpw_solver.cu` (`build_energy_device` MG branch) |
| **Function** | mid-SCF MG: **host collint** `multigrid_collocate_rho` + `ks_grid_build_rks` + `multigrid_integrate_veff` → device `H=H0+V` / DIIS (default). Device monom multi-level under-corrects Si (~4e-4 Ha, GH200 1180025). |
| **Metric** | both SCF converge; \|E_cuda − E_cpu\| ≤ **5×10⁻⁵ Ha** on diamond Si Γ ngrids=4 |
| **Not** | soft-tol; MG F/σ; multi-k MG; claiming full device-only monom MG green |

Parents already closed: PBC collocate; Si KB H0. This slug greens Si MG
**energy path** via collint twin (product-si host-cbg pattern). Residual:
device monom multi-level restore (env `ATV_GPW_CUDA_DEVICE_MG=1`).

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
