# Reference — CUDA PW non-default stream gate

## Purpose

Re-anchors the public `cuda::pw::PW` LSDA Si SCF after the density and band
sides share one non-default CUDA stream (slug `cuda-pw-nondefault-stream`).
**No physics change.** Pins and tolerances are identical to
[`../pw-cuda-lda-si/REFERENCE.md`](../pw-cuda-lda-si/REFERENCE.md).

## Oracle

| Gate | Source | Bound |
| --- | --- | --- |
| Literature window | PseudoDojo/ABINIT Si.djrepo \(E_\mathrm{min}\) | \([-16.916316, -15.416316]\) Ry |
| CPU↔CUDA parity | same-settings `cpu::pw::PW` | \(\|ΔE\| < 10^{-3}\) Ry |

`oracle: literature` (ABINIT-class) + same-settings CPU twin. Not self-golden.

## Stream contract (not a pin)

Production path: `DeviceDensity` creates a non-default stream; `PW::run`
passes it into `DeviceBandSolver`. Unit test
`CudaPWDensity.NonDefaultStreamOwnedAndShareable` asserts the handles.

## GH200 evidence

| Job | Result |
| --- | --- |
| **1049681** | 9/9 ctest; MAX_EIG_DIFF 5.274e-15 Ha; both regs \|ΔE\|=1.243e-14 Ry; `GPU_JOB_OK` |
