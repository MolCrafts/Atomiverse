# cuda-gpw-uks-payload-device

method=cuda | track=cuda | P3 | parent device-payload / uks-device-diis

## Lever

- **file+fn:** `src/cuda/gpw/gpw_solver.cu` final Γ UKS SCF block after
  DIIS loop; dual call to `AoDiisEigenDevice::eigen_from_device(d_Ha/d_Hb)`
- **metric:** remove host dual `diagonalize_fock` in former
  `store_converged_state_uks`; mid-path scientific matrix DH remains zero;
  boundary D2H only of dual C/ε/H_ks + P

## Acceptance

1. GH200 dual-gate `reg_cuda-gpw-uks-payload-device` PASS:
   - \|E_cuda − E_cpu\| ≤ 5×10⁻⁶ Ha
   - max\|ε_α/β\| ≤ 5×10⁻⁵ Ha
2. Grep own diff: zero mid-path scientific DeviceToHost/HostToDevice of
   ρ/V/H/P/C matrices (scalar residual + boundary payload OK)
3. Skills named: av-perf; write-numerically-stable-kernel;
   optimize-global-memory-access
4. **perf_exempt** (small-cell residency; no fair-wall claim)

## Out of scope

UKS F/σ, UKS-PBE, multi-k payload, cold-start, multigrid monom.
