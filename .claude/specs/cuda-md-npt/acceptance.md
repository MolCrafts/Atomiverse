---
spec: cuda-md-npt
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Device MTK NPT with CPU-identical public shape
    type: code
    pass_when: |
      cuda::md::NoseHooverNPT mirrors the CPU ctor exactly; unsupported
      knobs throw; no per-atom Verlet fallback introduced anywhere.
    status: pending
  - id: ac-002
    summary: fp64 single-step parity vs CPU
    type: runtime
    pass_when: |
      Unit tests show one-step integrator parity vs CPU within the stated
      tolerance at double precision on GH200.
    status: pending
  - id: ac-003
    summary: External anchor on the CUDA leg
    type: runtime
    pass_when: |
      The LAMMPS-pinned NPT regression passes on --device=cuda against the
      same external number as the CPU leg (not CPU-twin only).
    status: pending
out_of_scope:
  - PR full-cell / NPγT on device
  - multi-GPU
---

# Acceptance — `cuda-md-npt`

**Done** means CUDA NPT is device-principal, externally anchored, and
shape-identical to CPU.
