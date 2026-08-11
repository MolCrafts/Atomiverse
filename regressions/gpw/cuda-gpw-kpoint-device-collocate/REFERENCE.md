# cuda-gpw-kpoint-device-collocate — multi-k device χ^k density collocate

## What

Depth on public `ATV::cuda::gpw::GpwDft` multi-k RKS DIIS: mid-SCF density is
built **device-side** from uploaded χ^k + P_k via
`internal::KpointCollocateDevice` (twin of host `collocate_density_k`), then
`KsChainDevice::run_from_rho` with ρ already resident. Drops host
`collocate_density_k` + mid-SCF H2D of full ρ.

Host residuals (out of scope): `integrate_potential_k` (→
`kpoint-device-integrate`), `KDiis`, phased STVnl, complex Hermitian
diagonalize.

## Twin

Same-settings CPU `cpu::gpw::GpwDft` on H₂ (DZV-GTH-PADE / GTH-PADE-q1),
`k_mesh={2,1,1}` with **`k_time_reversal=false`** (nk=2), periodic 12 Bohr
cubic, CUTOFF 80 Ry, closed-shell RKS, Padé LDA, DIIS, single-grid,
`local_pp_on_grid=true`. Oracle: **CPU twin**.

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Both SCF converged; nk ≥ 2 | required |

## Perf

**perf_exempt** — small multi-k foothold (device spine lift); not a fair twin
wall cut vs CP2K.

## Non-goals

Forces/stress, UKS×k, MG×k, device integrate / KDiis / STVnl, production
Si/water dual-gates.

## GH200 evidence

**Post-rebase re-verify** job **1130716** (partition gpu, n22 GH200, module
`GPU/buildenv-gcccuda/2026.03-cu13.0`, tip `778d2186` on
`wf/cuda-gpw-kpoint-device-collocate` rebased onto `dev` @ `83ab8a5b`):

| Quantity | Value |
| --- | ---: |
| \|E_cuda − E_cpu\| | **3.353×10⁻¹⁴ Ha** |
| nk | 2 (TR-off 2×1×1) |
| CPU / CUDA iters | 5 / 5 |
| Unit `test_cuda_gpw_kpoint_collocate` | **Passed 1.84 s** |
| Unit `test_cuda_gpw_kpoints` (hold) | **Passed 0.93 s** |
| Parent `reg_cuda-gpw-kpoints` hold | \|dE\|=**3.353×10⁻¹⁴ Ha** |
| Exit | **GPU_JOB_OK** / sacct **COMPLETED 0:0** (~8m21s) |

Prior tip evidence (pre-rebase): job **1130429** n196, \|dE\|=**3.442×10⁻¹⁴ Ha**.

Tols: energy 5×10⁻⁶ Ha. **perf_exempt** small multi-k foothold.
