# cuda-gpw-kpoint-device-stvnl

## Lever

| | |
| --- | --- |
| **File** | `src/cuda/gpw/internal/kpoint_stvnl.cu` + multi-k branch of `src/cuda/gpw/gpw_solver.cu` (`run_cuda_kpoints_rks`) |
| **Function** | `KpointStvnlDevice::compute` — device phased contracted \(S_k/T_k\) (+ \(V_{\mathrm{nl}}^k\) when channels present); product no longer calls host `build_overlap_kinetic_k` / `add_nonlocal_kb_k` |
| **Metric** | multi-k RKS energy dual-gate \|E_cuda − E_cpu\| (H₂ k=2×1×1 PADE) |

## Residual closed

Multi-k product tip built **host** phased \(S_k/T_k/V_{\mathrm{nl}}^k\) once per `run` (Γ `device-stvnl` does not own multi-k lattice sums). After this slug: device Hermite / s–s contracted ST image sweep + Bloch phase accumulate; Vnl device-zero when no nonlocal channels (H GTH-PADE-q1); host KB residual only if atoms expose nonlocal channels (Si-class depth child).

## Acceptance

- Unit: `test_cuda_gpw_kpoint_stvnl` — device \(S_k/T_k\) vs host twin (max\|Δ\| stated)
- Regression: `reg_cuda-gpw-kpoint-device-stvnl` — product dual-gate \|dE\| ≤ 5e-6 Ha; both converged; nk≥2
- GH200 log evidence; **perf_exempt** (setup foothold, not fair-wall)
- Does **not** own KDiis, mid-SCF H assemble, \(P_k\) residency, payload, cold-start

## Oracle

Same-settings CPU `cpu::gpw::GpwDft` multi-k energy (not CP2K; not self-golden).
