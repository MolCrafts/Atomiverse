---
spec: cuda-md-lj-bench
created: 2026-08-14
criteria:
  - id: ac-001
    summary: Standing benchmark exists and runs on GH200
    type: runtime
    pass_when: |
      benchmarks/lj builds under ATV_BUILD_BENCHMARKS and a recorded GH200
      run produces walltime numbers for CPU-xsimd and CUDA at the A/B sizes.
    status: pending
  - id: ac-002
    summary: LAMMPS twin measured, provenance recorded
    type: docs
    pass_when: |
      The LAMMPS comparison run is recorded as binary version + input deck +
      printed output line (oracle-pin law); ratio computed from raw outputs.
    status: pending
  - id: ac-003
    summary: Roadmap ⚡ evidence is live
    type: docs
    pass_when: |
      force-fields.md points at the restored suite; ⚡ kept only if the
      measured ratio supports it, else downgraded with the number shown.
    status: pending
out_of_scope:
  - perf assertions in ctest
---

# Acceptance — `cuda-md-lj-bench`

**Done** means the repo's one perf claim is backed by a runnable benchmark
and a provenance-recorded external twin.
