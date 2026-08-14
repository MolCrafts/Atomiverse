---
slug: cuda-pw-spin-pbe
status: approved
created: 2026-08-14
domain: cuda / planewave
queue: claim-completion
deferred: true
depends_on: []
unblocks: [cuda-pw-spin-pbe]
grilled: false
---

# Spec: `cuda-pw-spin-pbe` — device spin-PBE forces/stress

**Deferred:** GH200 + CUDA ES open gate. Continues the sanctioned PW device
envelope (`.claude/notes/cuda-pw-large-cell.md` context).

## Summary

The CUDA PW backend computes collinear LSDA spin forces + stress and
restricted metallic-smear forces + stress, but **spin-PBE forces/stress
still reject**, as does spin+smear. Lift the two rejections: spin-resolved
PBE GGA gradient terms in the device XC/force/stress path, then the
spin×smear joint.

## Design

- Extend the device XC kernels with the spin-PBE gradient-correction terms
  (σ-resolved ∇ρ channels), mirroring the CPU spin-GGA implementation as
  the correctness oracle (same-settings CPU twin per ES policy).
- Stress: spin-GGA XC contribution on device; keep the established pattern
  of the closed-shell PBE stress kernels.
- Remove the `std::invalid_argument` rejections only when the numbers pass
  the dual gate — the loud refusal is the contract until then.
- Anchor: CPU twin (primary) + the QE-class external pin the CPU spin cases
  use (claim bar 3 — CUDA leg hits the external number).

## Files

- `src/cuda/planewave/…` XC/force/stress kernels + envelope checks
- `tests/cuda/planewave/` parity units (spin channels vs CPU)
- `regressions/pw/` CUDA leg on existing spin-PBE cases (multi-device)
- `tools/gpu_job_cuda_pw_spin_pbe.sh`

## Tasks

- [ ] Spin-PBE device XC energy/potential parity vs CPU
- [ ] Spin forces, then spin stress; lift rejections stepwise
- [ ] Spin×smear joint after both parents are green
- [ ] Dual-gated regression legs on GH200; roadmap PW CUDA column update

## Testing

Per-channel parity units (V_xc↑/↓, F, σ) vs CPU at stated tolerances;
regression dual gates with the CPU case's external anchor.

## Out of scope

- USPP/PAW, device bands/DOS, device DIIS/MT Poisson (separate residuals)
- Multi-atom smear-stress residual (`smear-stress-al2-gate`) unless it
  falls out naturally
