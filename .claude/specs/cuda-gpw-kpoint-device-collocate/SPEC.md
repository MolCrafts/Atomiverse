# cuda-gpw-kpoint-device-collocate

method=cuda | twin: same-settings CPU multi-k RKS energy dual-gate (H₂ k-class)

## Lever

| Field | Value |
| --- | --- |
| **File** | `src/cuda/gpw/gpw_solver.cu` (`run_cuda_kpoints_rks`) + new `src/cuda/gpw/internal/kpoint_collocate.cu` |
| **Function** | `KpointCollocateDevice::collocate_k` / mid-SCF `build_k_energy` ρ path |
| **Metric** | mid-SCF ρ built device-side from χ^k + P_k; no host `collocate_density_k` + H2D full ρ |

Multi-k product `run` previously **host** χ^k collocate + H2D ρ +
`KsChainDevice::run_from_rho` while Γ product owns device product collocate.
Compose device density collocate into multi-k `run` so ρ is built device-side
(foothold multi-k energy ≠ device multi-k KS spine).

Does **not** own: host `integrate_potential_k` (`kpoint-device-integrate`),
host `KDiis` (`kpoint-device-diis`), phased STVnl (`kpoint-device-stvnl`),
forces/stress, production Si/water dual-gates.

## Acceptance

- GH200 CUDA↔CPU multi-k RKS energy dual-gate with stated \|dE\| ≤ **5e-6 Ha**
- device-only collocate kernels on multi-k mid-SCF ρ (no host χ^k collocate
  for mid-SCF ρ); **perf_exempt** small cell
- unit: device ρ vs host `collocate_density_k` on fixed-P fragment
- no soft tols; same-settings CPU oracle; no dual-gate matrix expansion
- spin/MG multi-k stays reject if still ungated
