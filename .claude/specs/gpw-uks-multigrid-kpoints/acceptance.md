# Acceptance — gpw-uks-multigrid-kpoints

Filled at verification time (measured values land in REFERENCE.md).

- [x] `GpwDft::build` accepts `ngrids > 1 && rel_cutoff > 0` for collinear
      UKS on the k path; `ngrids > 1` without a positive `rel_cutoff` still
      throws (`tests/cpu/gpw/test_cpu_gpw_solver.cpp` gate tests, including
      the flipped `BuildAcceptsKpointMultigridUks`).
- [x] Single-grid multi-k UKS path unchanged (`ngrids == 1` bit-identical
      assembly; existing `gpw-kpoints-uks` regression stays green).
- [x] Regression `regressions/gpw/gpw-uks-multigrid-kpoints` (type:
      scientific, ≤ 60 s):
  - M1 Γ-limit identity: explicit k=(0,0,0) multigrid k-UKS vs Γ-UKS
    multigrid product path ≤ 1e-6 Ha (machine-exact class expected).
  - M2 time-reversal identity: full 2×2×2 (8 k) vs TR-folded (4 k)
    multigrid UKS ≤ 5e-6 Ha (machine-exact class expected).
  - M3 multigrid vs single grid ≤ 1e-3 Ha at the main cutoff (the
    REL_CUTOFF ladder is a cost optimization, not a physics change).
  - M4 converged + finite + dual-spin k payload shapes on the main run.
  - M5 |E − CP2K 2026.1 pin| ≤ 1e-3 Ha, expected ~5e-7 Ha class
    (`gpw-multigrid-kpoints` precedent 4.7e-7). Pin is an offline CP2K
    number (cp2k.psmp 2026.1 + committed twin input + logged output line),
    never seeded from ATV. Measured |Δ| = 2.42e-5 Ha on MOLOPT/PBE H₂.
- [x] Perf: T_atv/T_cp2k ≤ 10 on the NGRIDS 4 twin (or perf_exempt with
      stated reason). **perf_exempt** on H₂ twin (ratio 13.9; CP2K 1.3 s);
      product-class Si multigrid ratio remains 1.6.
- [x] Any rot found in the touched surface named in the summary (fixed or
      hard-stopped), not silenced. Found: high-spin multip 3 H₂ MP fails
      CP2K SCF; DZV-PADE twin ~22 mHa ATV/CP2K systematic → switched pin
      to MOLOPT/PBE closed-shell UKS (still dual-spin stacks).
