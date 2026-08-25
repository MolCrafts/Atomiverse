# Acceptance — `gfn2-xtb-water`

## Gates

1. Unit: `test_cpu_gfn2_params`
   - Static tables cover Z = 1, 6, 7, 8
   - Unsupported Z throws naming the supported set
   - H₂O SCC converges (`last_scc_residual() < scc_tol`)
   - Energy finite and nonzero
2. Regression: `run_reg_gfn2-xtb-water`
   - Public `cpu::Evaluate` + `cpu::semiempirical::GFN2`
   - Hard-coded pin vs xtb 6.7.1 **−5.070383095219 Eh** converted with
     `units::kHartreeInEv`
   - Tolerance **2×10⁻³ eV**: quadrupole AES and CN-function remainder
     (repulsion is 1 µEh vs the transcript)
3. `REFERENCE.md` states `oracle: xtb` and `oracle_version: 6.7.1 (edcfbbe)`
4. No live xtb / tblite / libxtb in ctest
5. SE-only cmake (CLI=OFF). No `gfn2_xtb_kernel.cpp`, no `ATV_HAS_XTB_KERNEL`

## Perf

Fair wall vs xtb 6.7.1 on this 3-atom single-point: target
\(T_{\mathrm{atv}}/T_{\mathrm{oracle}} \le 10\). xtb wall on the archived
run is 0.155 s (SCC 0.115 s). If the host has no xtb binary, mark
`perf_exempt` with that reason.

## Done when

`acceptance_met=true` in the worker schema and the regression prints `PASS`.
