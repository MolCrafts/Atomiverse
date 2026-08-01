# Acceptance — `gpw-d3-kpoint-forces`

| Check | Criterion | Status |
| ----- | --------- | ------ |
| Capability | multi-k `assemble_gpw_forces_kpoints` folds D3 zero \(F_{\mathrm{disp}}\) | present on tip |
| Oracle twin | ENERGY_FORCE + MP 1×1×2 D3 on/off inputs under `tools/cp2k_oracle/inputs/gpw-d3-kpoint-forces/` | written |
| Real CP2K pin | binary+version+input+output lines in `.note` / REFERENCE | run 2026-08-01, git:5e54ba2 |
| Regression dual gate | E_disp, F_disp Δ, multi-k wiring, wall ≤ 60 s | `reg_gpw-d3-kpoint-forces` |
| Fair wall | \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\) vs D3 multi-k ENERGY_FORCE | \(T_{\mathrm{cp2k}}=3.741\) s |
| `perf_exempt` | false | required |
| `oracle` | cp2k | not literature |

`acceptance_met` = **true** after scoped build + `run_reg_gpw-d3-kpoint-forces`
green (2026-08-01): max\|F_disp−cp2k\|=7.858e−07, wiring=1.3e−17,
ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}=4.353\) (16.284 / 3.741).
