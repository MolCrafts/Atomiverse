# Acceptance — gpw-metallic-stress

| Gate | Criterion |
| ---- | --------- |
| Unit | `test_cpu_gpw_stress` metallic-smear analytic vs multi-SCF FD \(\le 5\times10^{-5}\) Ha/Bohr³ on H₂ fixture |
| Regression | `reg_gpw-metallic-stress` dual-gate analytic vs FD \(\le 2\times10^{-4}\) Ha/Bohr³ on BCC Na FD |
| Payload | SCF exposes non-empty `f_k` and full `C_k` under smear; \(N_e\) pin |
| Oracle | Real CP2K 2026.1 twin run recorded in REFERENCE.md (binary + version + input + stress line) |
| Perf | Fair wall \(T_{\mathrm{atv,scf}}/T_{\mathrm{cp2k}}\) on same twin SCF job \(\le 10\) |
| API | No public `*Config`; no `make_*` / `build_*` factories; Frame I/O only for drivers |

`acceptance_met=true` only when all gates green with offline CP2K pin.
