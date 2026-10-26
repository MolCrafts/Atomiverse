# CUDA GPW — target public shape and remaining debt

Target contract for `cuda::gpw::GpwDft`, not a claim of current conformance.
Capability/evidence status: [DFT roadmap](../../docs/roadmap/dft.md).
Residency law:
[cuda-only-no-dh.md](cuda-only-no-dh.md).

## Frozen shape (must equal CPU `GpwDft`)

```text
cuda::gpw::GpwDft method{kinds, cutoff_ry, xc};  // identity: ctor args
method.tol = …;                    // knobs: members, not *Config
method.build(frame);               // cell from frame.box, Z → kinds
auto& res = method.run(frame);     // eV on kEnergy; Hartree on GpwResult
method.forces(frame);              // eV/Å on kAtom_f*
const auto& last = method.last_result();
method.stress(frame);              // target: same member shape as CPU
```

- Concrete method under `ATV::cuda::gpw` (no freestanding `SCFSolver`);
  identity (`kinds`, `cutoff_ry`, `xc`) as ctor args, knobs as members; no
  `*Config`, no `make_*`.
- Target: `build(frame)` fail-fast; re-entrant `run` / `forces`; `last_result`;
  geometry never a member (no public `cell`) and never an atom list
  (`internal::GpwAtom` is engine-private). Same `GpwResult` / `XcKind` as CPU.
- Stage kernels stay under `cuda/gpw/internal/` (`KsChainDevice`,
  `KsMultiGridChainDevice`, `PoissonFftDevice`, product collocate/integrate,
  spectral MG transfer, XC grids) and are composed, never promoted to user API.
- Capability subset is allowed; a public-shape fork is not. Unsupported options
  (OT, MT, …) fail closed. CPU OT free-loop residual does not block CUDA.
- No architecture-freeze regression is registered. Review conformance against
  the actual public header; physical cases live in `regressions/gpw/{rks,uks}`.

## Product state

- Frame `build` / `run` / `forces` exist and read `kinds` + `frame.box`, but
  `include/atomiverse/cuda/gpw/gpw_solver.cuh` still publicly exposes the
  `GpwAtom` alias, `cell[3]`, atom-list build/run and free force/stress entries.
  The Frame `stress` member is missing (`es-cuda-gpw-frame-io`). The SCF is a
  free-loop composer inside `run()`, not
  a nested `cuda::Driver<gpw::Phase>` — fold only shape-preservingly.
- Mid-SCF on device (non-default stream): P resident → weight pack → ρ
  collocate (SG PBC sphere-window | MG monom multi-level) → Poisson + XC →
  V_eff → integrate → V_AO scatter → H = H0 + V → device DIIS residual → eigen
  → P (`AoDiisEigenDevice`, `KpointDiis`). Final eigen payload D2H once.

| Path | Mid-SCF on device |
| --- | --- |
| Γ RKS single-grid | yes |
| Γ RKS multigrid | yes (monom default; `ATV_GPW_CUDA_HOST_MG=1` is debug only) |
| Γ UKS single-grid | yes (dual pack / H / DIIS / payload) |
| Multi-k RKS | mid-SCF yes; setup χ / STVnl hybrid |

- Forces: device grid Hellmann–Feynman + **host AO Pulay** residual; stress via
  a host assembly shell.
- Evidence class is CPU-twin (◐), not CP2K; promoting to ✓ needs independent
  CP2K pins.

## Open depth (do not soft-land)

1. `device-cold-start` — densify still on host.
2. Multi-k χ / orth / Vnl / H0 residency, payload and cold-start.
3. Device-native AO Pulay / force handoff; device stress kernels.
4. Production multi-k / MG / UKS × PBE joints; Si MG F/σ.
5. Nested `cuda::Driver<gpw::Phase>` (shape-preserving fold only).

## Forbidden

A public CUDA GPW API that differs from CPU `GpwDft` (Config bag, free-loop
solver type, foothold-as-product); promoting `KsChainDevice` to user surface;
softened or self-golden CUDA↔CPU pins; claiming green without GH200 evidence.
