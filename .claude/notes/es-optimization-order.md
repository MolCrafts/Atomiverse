<!-- mol:note:topic:es-optimization-order -->
## ES optimization order — current state

Binding law: CLAUDE.md › Electronic structure optimization order. This page
holds only the current focus; rewrite in place.

### CPU (step 2: performance)

- GPW structure is single-owner: private `GpwWorkspace`, shared
  `ks_grid_build_rks/uks`, CP2K-style collint with geometry batching and
  shell-level tasks, multi-level MAIN composition (`multigrid_collocate_rho*` /
  `multigrid_integrate_veff*`) for Γ and multi-k. OT economics, g·e DIIS outer
  and OUTER_SCF polish are in.
- `cpu_pending = 0`: no pin-safe wall lever is left. The remaining ratio (MG
  MAIN, [gpw-perf-baseline.md](gpw-perf-baseline.md)) needs a non-pin-safe
  work reduction on the shared MAIN path (`mg-main-structural`), and its
  baseline must first be re-measured without OpenMP binding.

### CUDA (step 3: method slices)

- Landed slices and their envelopes: [cuda-rhf-device-only.md](cuda-rhf-device-only.md)
  (RHF, PW), [cuda-gpw-public-shape.md](cuda-gpw-public-shape.md) (GPW).
  Residency law: [cuda-only-no-dh.md](cuda-only-no-dh.md).
- Next depth (queue `.claude/campaigns/es-cpu-cuda-queue.md`):
  - GPW: `device-cold-start`, multi-k setup residency (χ / Löwdin / KB Vnl /
    H0) and payload/cold-start, device AO Pulay and force handoff, device
    stress, `multigrid-si-gate`, production multi-k / MG / UKS × PBE F/σ.
  - PW: spin×smear stress, spin multi-k, spin×kmesh×smear joints.
  - Nested `cuda::Driver<gpw::Phase>` only as a shape-preserving fold.

### Forbidden

No hybrid / GAPW / PAW / meta-GGA product in this track; no soft tolerances or
self-golden pins; no CUDA public shape that differs from CPU; global
`XTENSOR_USE_XSIMD` and arch intrinsics; leaving multi-GB builds behind; nested
long-lived subagents; inventing capability gaps here (method loops own them).
