# Acceptance — cuda-gpw-uks-o2-gate

| Gate | Criterion |
| --- | --- |
| Dual-gate energy | \|E_cuda − E_cpu\| ≤ **5×10⁻⁵ Ha** (composite: cuFFT/FFTW + p-shell product tree; not bit-exact) |
| SCF | both CPU and CUDA converged |
| Spin identity | CUDA ⟨S²⟩ > 1.5 (multiplicity-3 O₂) |
| Result payloads | CUDA result has veff_a/veff_b/v_es and P_beta when n_beta>0 |
| Unit | `test_cuda_gpw_full_product` includes O₂ UKS dual-gate (GPU) |
| Regression | `reg_cuda-gpw-uks-o2-gate` green on GH200 with log evidence |
| No new kernel | only unit/reg + CMake; product reuses `run_uks` |
| Disk | build* wiped after gate; no leftover build dirs |

`acceptance_met=true` only with GH200 dual-gate PASS in the job log.
