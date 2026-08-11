# cuda-gpw-uks-device-diis

## Twin

Same-settings CPU `GpwDft` high-spin H₂ Γ UKS PADE energy after dual-spin
mid-SCF Pulay DIIS + dense eigen move onto device (two `AoDiisEigenDevice`
channels, `occ_factor=1`). Geometry and settings match `cuda-gpw-uks-gamma`
energy gate (foothold proving the device SCF spine, not a new chemistry).

## Oracle

`oracle: cpu-twin` — same-settings `cpu::gpw::GpwDft` energy on the dual-gate
binary (not a literature pin).

## Tolerances

| Quantity | Tol |
| --- | ---: |
| \|E_cuda − E_cpu\| | ≤ 5×10⁻⁶ Ha |

**perf_exempt** — small-cell architectural lift (device UKS DIIS/eigen), not a
fair-twin wall measurement.

## Not owned

Free-fn UKS forces/stress, UKS-PBE, multi-k UKS, device weight-pack.

## Evidence

| Job | Node | Result |
| --- | --- | --- |
| GH200 **1130279** | n24 | `test_cuda_gpw_diis_eigen` PASS; \|dE\|=**4.760e-06** Ha (tol 5e-6); uks-gamma smoke PASS same \|dE\| |
| GH200 **1130402** | n502 | re-verify on wave-8 parent `dev` @ `01bfcf9b` (feat tip `6df76cf7`); unit PASS; \|dE\|=**4.760e-06** Ha; uks-gamma smoke PASS same \|dE\|; `GPU_JOB_OK` |
