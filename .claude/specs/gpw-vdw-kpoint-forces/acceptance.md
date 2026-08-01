# Acceptance — gpw-vdw-kpoint-forces

- [x] Spec present under `.claude/specs/gpw-vdw-kpoint-forces/`
- [x] Offline CP2K 2026.1 ENERGY_FORCE twin run (NON_LOCAL DRSLL + KPOINTS);
      binary + git sha + input + key output lines recorded in REFERENCE.md
- [x] Regression `regressions/gpw/gpw-vdw-kpoint-forces/` dual-gates:
      - E_disp (and E_tot within twin-family systematic) vs CP2K pin
      - Analytic multi-k + vdW forces vs CP2K FORCES; offline FD identity
- [x] Fair wall \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\) on the matched twin
      (measured ~1.6–3.3; not multi-gate reg wall; not `perf_exempt`)
- [x] CMake: `atv_add_regression(gpw gpw-vdw-kpoint-forces …)`
- [x] Scoped build+test green; commit on `wf/gpw-vdw-kpoint-forces`
- [x] No live CP2K in ctest; no public Config/factories; Frame-only I/O

`acceptance_met = true`.
