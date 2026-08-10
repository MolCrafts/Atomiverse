# Acceptance — cuda-gpw-ks-gga

- [ ] `SpectralFftDevice` gradient/divergence match CPU `grid_fft` on a small
      orthorhombic field (unit, optional self-check inside GGA test).
- [ ] `KsChainDevice::run(..., KsXcKind::PBE)` produces V_eff / e_H / e_xc /
      product acc matching `ks_grid_build_rks(..., XcKind::PBE)` within
      stated stage tols (energy rel ~1e-10; V_eff ~1e-9·peak + 1e-12;
      acc same class as LDA chain).
- [ ] Unit `test_cuda_gpw_ks_gga` PASS on GH200.
- [ ] Dual-gate `reg_cuda-gpw-ks-gga` PASS on GH200 with log evidence.
- [ ] Existing LDA `test_cuda_gpw_ks_chain` still green (API default PADE).
- [ ] No public `GpwDft` surface; device-only scientific loops in `.cu`.
- [ ] Build wiped after test; no leftover `build*`.

`acceptance_met=true` only with GH200 log path in the worker summary.
