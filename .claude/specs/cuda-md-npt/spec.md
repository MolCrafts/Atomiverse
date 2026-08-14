---
slug: cuda-md-npt
status: approved
created: 2026-08-14
domain: cuda / classical MD
queue: claim-completion
deferred: true
depends_on: []
unblocks: [cuda-md-npt]
grilled: false
---

# Spec: `cuda-md-npt` — device NPT for the CUDA MD engine

**Deferred:** GH200 batch work. Classical-MD CUDA track (independent of ES
freezes). The OpenMM tile-engine architecture rule is binding — no per-atom
Verlet fallback, no NonbondedStyle dual architecture.

## Summary

CUDA MD claims NVE/NVT/NPT parity with CPU in the roadmap ensembles table,
but CUDA NPH/PR are ○ and the device barostat story is thinner than CPU.
Deliver MTK NoseHooverNPT semantics on the CUDA backend: device
virial/stress input, cell scaling on device, no per-step D2H of state.

## Design

- Mirror the CPU `cpu::md::NoseHooverNPT` public shape exactly
  (`cuda::md::NoseHooverNPT{T, tau_T, P, tau_P, dt}` as Integrator) — same
  ctor, same defaults; a CPU-only knob that cannot be honored on device
  throws, never silently ignores.
- Virial: the tile force kernels already accumulate stress for LJ/Coulomb
  (verify; extend where missing). Cell update + rescale on device
  (`cuda::Box`), neighbor rebuild triggered by box change.
- Anchor per claim bar 3: CPU-twin trajectory tolerance (stated) **plus**
  the LAMMPS NPT pin the CPU case uses — the CUDA leg must hit the same
  external number, not only the CPU leg.

## Files

- `include/atomiverse/cuda/dynamics/nose_hoover_npt.cuh`,
  `src/cuda/dynamics/nose_hoover_npt.cu`
- device stress plumbing where missing (`cuda/potentials/*`)
- `tests/cuda/dynamics/…` unit tests (single-step parity vs CPU at fp64)
- `regressions/md/` CUDA leg on the existing NPT case (multi-device
  PyTorch-style `--device`, house pattern)
- `tools/gpu_job_cuda_md_npt.sh`

## Tasks

- [ ] Audit device virial coverage for the bound styles; fill gaps
- [ ] Implement device MTK NPT integrator + cell/neighbor interaction
- [ ] Single-step fp64 parity unit tests vs CPU (stated tolerance)
- [ ] Extend the existing LAMMPS-pinned NPT regression with a CUDA leg;
      dual-gate green on GH200
- [ ] Roadmap ensembles table CUDA column update

## Testing

- Unit: one-step integrator parity vs CPU at double precision.
- Regression: external LAMMPS pin hit by the CUDA leg (claim bar 3).

## Out of scope

- Parrinello–Rahman full-cell on device, NPγT
- CUDA constraints (RATTLE/SETTLE device ports)
- Multi-GPU NPT
