# cuda-gpw-kpoint-device-integrate

method=cuda | twin: same-settings CPU multi-k RKS energy dual-gate (H₂ k-class)

## Lever

| | |
| --- | --- |
| File | `src/cuda/gpw/internal/kpoint_integrate.cu` + multi-k branch of `src/cuda/gpw/gpw_solver.cu` (`run_cuda_kpoints_rks` / `build_k_energy`) |
| Function | `KpointIntegrateDevice::integrate` ← device twin of host `integrate_potential_k` |
| Metric | mid-SCF: V_eff stays device-resident; only nb×nb complex Vmat D2H per k; dual-gate \|dE\| |

Multi-k product tip previously **D2H full V_eff** + host `integrate_potential_k`
every SCF iter after device `run_from_rho`. Density collocate remains host
(`kpoint-device-collocate`). This slug owns the integrate half only.

Does **not** own: KDiis (`kpoint-device-diis`), phased STVnl
(`kpoint-device-stvnl`), forces/stress, device collocate.

## Acceptance

- GH200 CUDA↔CPU multi-k RKS energy dual-gate with stated \|dE\| ≤ 5e-6 Ha
- no mid-SCF D2H V_eff + host integrate for the H_k path; **perf_exempt** small cell
- unit: device Vmat vs host `integrate_potential_k` on synthetic V
- no soft tols; same-settings CPU oracle; device-only kernels in `.cu`
- does not expand dual-gate matrix (Si/water F/σ stay child rows)
