---
spec: cuda-scf-gradient
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Device-principal gradient for covered shells
    type: runtime
    pass_when: |
      cuda::scf::RHF::forces computes on device for l<=1 basis sets (no
      host fallback); out-of-coverage throws with a how-to.
    status: pending
  - id: ac-002
    summary: CPU and PySCF anchors both green
    type: runtime
    pass_when: |
      CPU-vs-CUDA force parity within stated tolerance, and the CUDA leg of
      a PySCF-pinned gradient regression passes on GH200.
    status: pending
  - id: ac-003
    summary: GPU AIMD exists
    type: runtime
    pass_when: |
      A short GPU BOMD smoke runs; roadmap "no GPU AIMD" statement replaced
      by the honest new scope.
    status: pending
out_of_scope:
  - l >= 2, UHF/KS device gradients
  - perf claims
---

# Acceptance — `cuda-scf-gradient`

**Done** means the GPU AIMD blocker is gone for the covered shell class,
with dual anchors and no silent host fallback.
