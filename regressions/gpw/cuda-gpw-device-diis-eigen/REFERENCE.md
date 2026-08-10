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
