---
spec: cuda-gpw-product
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Frozen-shape product matrix complete
    type: runtime
    pass_when: |
      The es-cpu-cuda queue items for the frozen CUDA GPW shape (water
      stress, multi-k forces/stress, UKS forces/stress, MG production) are
      dual-gate green on GH200.
    status: pending
  - id: ac-002
    summary: No CUDA GPW claim rests on a self-anchor
    type: docs
    pass_when: |
      Every cuda-gpw-*/cuda-pw-* case marked ✓ carries an external pin
      (CP2K twin per the strict law); CPU-twin-only cases remain ◐ and say
      so in REFERENCE.md.
    status: pending
  - id: ac-003
    summary: Inventory and roadmap synchronized
    type: docs
    pass_when: |
      cuda-gpw-product-inventory.md and the dft.md CUDA column reflect the
      landed matrix with honest ◐/✓ marks.
    status: pending
out_of_scope:
  - shape changes
  - hybrids/meta-GGA/GAPW
---

# Acceptance — `cuda-gpw-product`

**Done** means the frozen CUDA GPW contract is fully staffed and every ✓ in
its column is independently anchored.
