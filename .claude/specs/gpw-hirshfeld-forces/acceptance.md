# Acceptance — `gpw-hirshfeld-forces`

| Gate | Criterion | Met |
| ---- | --------- | --- |
| Capability | Forces/stress fold Hirshfeld volumes when `hirshfeld=true` | yes |
| Unit dual | `test_cpu_gpw_vol_forces` fixed-\(v\) FD + product wiring | yes |
| Regression | `reg_gpw-hirshfeld-forces` science gates + wall ≤ 60 s | yes |
| Oracle | literature/libMBD dual (no CP2K TS/MBD) | yes |
| Perf | `perf_exempt` (no matched CP2K twin) | yes |

`acceptance_met = true` when all rows yes.
