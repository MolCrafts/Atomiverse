# cuda-gpw-device-diis-eigen

## Twin

Same-settings CPU `GpwDft` H₂ Γ RKS PADE energy after RKS mid-SCF Pulay DIIS
+ dense eigen move onto device (`AoDiisEigenDevice` / cuSOLVER). Geometry and
settings match `cuda-gpw-full-product` energy gate.

## Oracle

`oracle: cpu-twin` — same-settings `cpu::gpw::GpwDft` energy on the dual-gate
binary (not a literature pin).

## Tolerances

| Quantity | Tol |
| --- | ---: |
| \|E_cuda − E_cpu\| | ≤ 5×10⁻⁶ Ha |

**perf_exempt** — small-cell architectural foothold (device DIIS/eigen), not a
fair-twin wall measurement.

## Evidence (wave-2 resume)

| Job | Node | Result |
| --- | --- | --- |
| GH200 **1129285** (orig tip) | n189 | unit + dual-gate PASS \|dE\|≈1.6e-7 Ha |
| GH200 **1129719** (rebase on `bcb873e0`) | n194 | `test_cuda_gpw_diis_eigen` PASS; \|dE\|=**1.616e-07** Ha |

Rebase: `84096a1e` → `e8170b10` onto wave-11 GapCensus parent.
