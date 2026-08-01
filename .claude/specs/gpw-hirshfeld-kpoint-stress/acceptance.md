---
spec: gpw-hirshfeld-kpoint-stress
created: 2026-08-01
criteria:
  - id: ac-001
    summary: Fixed-v MBD energy dual pin
    type: runtime
    pass_when: |
      reg_gpw-hirshfeld-kpoint-stress gate A: |E_disp MBD@rsSCS(fixed dual v)
      − pin| ≤ 1e-8 Ha (pin −1.1479717148086300e-03 Ha).
    status: pass
  - id: ac-002
    summary: Geometry stress dual under strain (TS + MBD fixed-v)
    type: runtime
    pass_when: |
      Gate B: product ts_pbe / mbd_rsscs_pbe with fixed dual volumes —
      central FD of E_disp under orthorhombic strain matches analytic
      σ_disp = −ΔE/(2hΩ) to ≤ 1e-12 Ha/Bohr³ (self dual of the same
      energy surface the stress Lagrangian rebuilds).
    status: pass
  - id: ac-003
    summary: Multi-k TS+Hirshfeld stress wiring
    type: runtime
    pass_when: |
      Gate C on multi-k H2 (k_mesh 1×1×2): σ(ts+hirsh)−σ(plain) equals
      geometry-only strain-FD of ts_pbe at scf.volume_ratios within
      1e-8 Ha/Bohr³; residual vs free-atom TS exceeds 1e-12.
    status: pass
  - id: ac-004
    summary: Multi-k MBD+Hirshfeld stress wiring
    type: runtime
    pass_when: |
      Gate D: same wiring for mbd_rsscs_pbe at scf.volume_ratios (isolated
      when mbd_k_mesh default zero); multi-k payload present.
    status: pass
  - id: ac-005
    summary: REFERENCE + oracle notes; no live CP2K in ctest
    type: docs
    pass_when: |
      REFERENCE.md and tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoint-stress/
      document oracle: literature/libMBD, perf_exempt, CP2K version check
      (no stock TS/MBD), and dual-gate tolerances. Regression never
      launches CP2K.
    status: pass
  - id: ac-006
    summary: Wall budget
    type: runtime
    pass_when: |
      Product regression wall ≤ 60 s on the regression host.
    status: pass
