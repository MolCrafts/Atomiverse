<!-- mol:note:topic:es-optimization-order -->
## ES optimization order — current state

Binding law: CLAUDE.md › Electronic structure optimization order. This page
holds only the current focus; rewrite in place.

### CPU (architecture closure and numerical recheck before performance)

- Reconcile remaining architecture work and surviving historical numerical
  reports using [es-architecture-debt.md](es-architecture-debt.md). Source
  closure of direct PW/BO GPW mounting is not numerical acceptance; GPW OT,
  propagator composition and public-consumer regressions remain open.

- GPW structure is single-owner: private `GpwWorkspace`, shared
  `ks_grid_build_rks/uks`, CP2K-style collint with geometry batching and
  shell-level tasks, multi-level MAIN composition (`multigrid_collocate_rho*` /
  `multigrid_integrate_veff*`) for Γ and multi-k. OT economics, g·e DIIS outer
  and OUTER_SCF polish are in.
- The prior pin-preserving kernel pass identified no further local lever.
  That does not mean CPU work is complete. First remeasure the twins in
  [gpw-perf-baseline.md](gpw-perf-baseline.md) without OpenMP binding, then
  reprofile before choosing structural work reduction on shared MAIN,
  products-default-k or MPI. The old ratios do not establish today's gap.

### CUDA (step 3: method slices)

- Landed slices and their envelopes: [cuda-rhf-device-only.md](cuda-rhf-device-only.md)
  (RHF, PW), [cuda-gpw-public-shape.md](cuda-gpw-public-shape.md) (GPW).
  Residency law: [cuda-only-no-dh.md](cuda-only-no-dh.md).
- Candidate depth (reconcile any generated queue with current source first):
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
