# Acceptance — `gpw-hirshfeld-uks-kpoints`

| # | Criterion | Gate |
| - | --------- | ---- |
| 1 | Multi-k UKS Hirshfeld product path exercises `fold_dispersion_k_uks` | unit + reg |
| 2 | A1 closed-shell multi-k UKS ≡ RKS E_disp ≤ 5e-6 Ha | reg |
| 3 | A2 O₂ multi-k UKS E_disp pin ≤ 1e-8 Ha | reg |
| 4 | Offline notes cite libMBD / literature (no fake `oracle:cp2k`) | `tools/cp2k_oracle/…` |
| 5 | Wall ≤ 60 s class; perf_exempt for missing CP2K TS/MBD twin | REFERENCE.md |

`acceptance_met = true` when all five pass on `wf/gpw-hirshfeld-uks-kpoints`.
