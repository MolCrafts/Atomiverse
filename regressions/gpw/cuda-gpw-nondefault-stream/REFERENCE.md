# cuda-gpw-nondefault-stream

## Twin

Same H₂ Γ RKS PADE dual-gate as `cuda-gpw-full-product` / device-forces after
product SCF moves mid-SCF transfers + KS chain onto a **non-default**
`cudaStream_t` owned for the duration of `GpwDft::run()`.

## Tolerances

Unchanged vs full-product: |ΔE| ≤ 5e-6 Ha, max|ΔF| ≤ 5e-5 Ha/Bohr.

**perf_exempt** small cell.
