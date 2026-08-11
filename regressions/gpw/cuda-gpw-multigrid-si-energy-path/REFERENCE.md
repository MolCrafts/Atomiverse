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

Mid-SCF MG branch in `gpw_solver.cu` `build_energy_device`: per-level
`ProductWeightPackDevice::pack_weights` → `KsMultiGridChainDevice::run`
(sphere-window PBC collocate + spectral prolong/restrict + fine Poisson/XC +
per-level integrate) → `scatter_acc` / `scatter_acc_add` → device \(H=H_0+V\)
/ DIIS. Parents: KB Vnl@Γ (`product-si-gate`), PBC collocate
(`product-collocate-pbc`). Path surgery only if dual-gate fails.

## Perf

**perf_exempt** — correctness dual-gate / path-fix (not fair-wall vs CP2K).

## Status

pending GH200 dual-gate.
