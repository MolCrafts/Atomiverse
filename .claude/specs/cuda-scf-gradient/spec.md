---
slug: cuda-scf-gradient
status: code-complete
created: 2026-08-14
domain: cuda / scf
queue: claim-completion
deferred: true
depends_on: []
unblocks: [cuda-scf-gradient, gpu-aimd]
grilled: false
---

# Spec: `cuda-scf-gradient` — device RHF gradients (unblocks GPU AIMD)

**Deferred:** GH200 + the **CUDA ES open gate** (CLAUDE.md): CPU pins
green, public `run`/`forces` shape stable, tolerances stated. Sequencing
note: the nested `cuda::Driver<scf::Phase>` fold residual
(`.claude/notes/cuda-rhf-device-only.md`) should land first or alongside —
do not build the gradient on the to-be-replaced `RhfScfCycle` shape without
checking that note's sanctioned wave order.

## Summary

"CUDA SCF has no gradient kernel — there is no GPU AIMD" (roadmap §7). The
CUDA RHF engine covers s/p (l ≤ 1) device-native integrals for energy;
deliver the matching analytic gradient kernels so `cuda::scf::RHF::forces
(frame)` is device-principal for the same shell coverage, and a GPU BOMD
smoke becomes possible.

## Design

- Derivative ERI + one-electron derivative kernels for l ≤ 1 shells,
  device-resident (CUDA programming style per ES policy — no giant
  dual-backend source; CPU is the correctness oracle).
- Same-settings CPU forces as the primary oracle; PySCF gradient pin on one
  molecule as the external anchor (claim bar 3).
- Public shape unchanged: `cuda::scf::RHF{basis}.forces(frame)` already
  exists as API — this fills it for real instead of host fallback (verify
  current behavior first; if it silently falls back to CPU today, that is
  a defect to name).
- Out-of-coverage shells throw the same how-to error the energy path uses.

## Files

- `src/cuda/scf/…` derivative kernels + `include/atomiverse/cuda/scf/…`
- `tests/cuda/scf/` force parity units (CPU vs CUDA, stated tol)
- `regressions/scf/` CUDA leg on an existing PySCF-pinned gradient case
- `tools/gpu_job_cuda_scf_gradient.sh`

## Tasks

- [x] Verify current `forces` behavior + wave order vs the cuda-rhf note
- [x] Derivative kernels (overlap/kinetic/nuclear/ERI, l ≤ 1) on device
- [x] CPU↔CUDA parity units at stated tolerance; PySCF-pinned regression leg
- [x] GPU BOMD smoke (few NVE steps, H₂/H₂O class); roadmap §7 update

## Testing

- Unit parity per operator (S′, T′, V′, ERI′) against CPU.
- Regression: existing molecular gradient pin, CUDA leg dual-gated.

## Out of scope

- l ≥ 2 shells, UHF/KS gradients on device
- CUDA GPW/PW gradients (other rows)
- Perf claims (correctness-only; small molecules)
