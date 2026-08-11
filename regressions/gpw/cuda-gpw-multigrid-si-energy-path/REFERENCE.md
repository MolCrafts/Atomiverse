# `cuda-gpw-multigrid-si-energy-path`

## What

Production **Si-class Γ RKS PADE multigrid** product energy path on public
`cuda::gpw::GpwDft` vs same-settings CPU. Closes the gap after H₂
`multigrid-run` foothold + Si single-grid `product-si-gate` + PBC collocate:
Si nonlocal × multi-level device KS continuous path.

## Twin

| | |
| --- | --- |
| System | diamond Si bct 4-atom, a = 5.43 Å |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| Settings | Γ, CUTOFF 40 Ry, **ngrids=4**, **rel_cutoff=20**, n_occ=8, PADE, DIIS |
| Oracle | **same-settings CPU** (not CP2K) |

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| Both SCF converged | required |

## Lever

Mid-SCF MG branch in `gpw_solver.cu` `build_energy_device`:

1. **Default (production path for this slug):** host collint multi-level twin
   `multigrid_collocate_rho` + `ks_grid_build_rks` + `multigrid_integrate_veff`
   (same MAIN as CPU), then device \(H=H_0+V\) / DIIS. Mid-path P D2H + V H2D
   residual (iron-law P2) — same pattern as product-si host cbg before
   `product-collocate-pbc`.
2. **Device monom residual:** `ATV_GPW_CUDA_DEVICE_MG=1` forces
   pack→`KsMultiGridChainDevice`→scatter. GH200 **1180025**: monom multi-level
   under-corrects Si (CUDA MG E ≈ SG −15.1523 vs CPU MG −15.15189,
   \|dE\|=**4.3e-4**). H₂ monom MG stays green (s-only). Deeper restore later.

Parents: KB Vnl@Γ (`product-si-gate`), PBC collocate (`product-collocate-pbc`).

## Perf

**perf_exempt** — correctness dual-gate / path-fix (not fair-wall vs CP2K).

## Status

**acceptance_met=true** — GH200 **1180409** (tip `a58d90e1`):
CPU E=−15.151890096614 Ha iters=9; CUDA E=−15.151890075311 Ha iters=9;
\|dE\|=**2.130e-08** Ha (tol 5e-5). Holds: ks_multigrid unit, full_product,
multigrid-run \|dE\|=**1.082e-11**, product-si \|dE\|=**3.504e-05**.
Prior fail **1180025** device monom \|dE\|=**4.296e-04**.
