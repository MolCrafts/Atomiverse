# Acceptance — cuda-gpw-uks-payload-device

| Check | Status |
| --- | --- |
| Spec lever stated before code | yes |
| Dual `eigen_from_device` for α/β final payload | yes |
| Host `store_converged_state_uks` re-diag removed | yes |
| Regression dual-gate registered | yes |
| GH200 green log | **1186414** n514 PASS |
| Mid-path scientific matrix DH = 0 in diff | yes |
| Skills named in summary | yes |
| `acceptance_met` | **true** |

## Tolerances

|dE| ≤ 5e-6 Ha; max|deps α/β| ≤ 5e-5 Ha; oracle=cpu-twin; perf_exempt.

## Measured (GH200 1186414)

|dE|=4.163e-09 Ha; max|deps|=1.481e-06 Ha.
