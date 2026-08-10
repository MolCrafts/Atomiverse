# Acceptance — cuda-gpw-ks-gga

- [x] `SpectralFftDevice` gradient/divergence match CPU `grid_fft` on a small
      orthorhombic field (unit inside `test_cuda_gpw_ks_gga`).
- [x] `KsChainDevice::run(..., KsXcKind::PBE)` produces V_eff / e_H / e_xc /
      product acc matching `ks_grid_build_rks(..., XcKind::PBE)` within
      stated stage tols (energy rel ~1e-10; V_eff ~1e-9·peak + 1e-12;
      acc same class as LDA chain).
- [x] Unit `test_cuda_gpw_ks_gga` PASS on GH200 (job **1050278**, 1.17 s).
- [x] Dual-gate `reg_cuda-gpw-ks-gga` PASS on GH200 with log evidence
      (`.claude/cuda-gpw-ks-gga-1050278.out`).
- [x] Existing LDA path still green via default `KsXcKind::PADE` unit case.
- [x] No public `GpwDft` surface; device-only scientific loops in `.cu`.
- [x] Build wiped after test; no leftover `build*`.

`acceptance_met=true` — GH200 job **1050278**.
