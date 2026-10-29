# CUDA GPW — target public shape and remaining debt

Public contract for `cuda::gpw::GpwDft`; source shape and scoped native
Driver acceptance now conform. Complete device residency and CPU capability
parity remain open.
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
auto sigma = method.stress(frame); // eV/Å³; kStress = sigma * volume [eV]
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
  (OT, MT, …) fail closed. CPU OT now uses its typed Driver; CUDA OT remains
  a separate numerical implementation task.
- No architecture-freeze regression is registered. Review conformance against
  the actual public header; physical cases live in `regressions/gpw/{rks,uks}`.

## Product state

- Frame `build` / `run` / `forces` / `stress` read `kinds` + `frame.box`.
  Atom payload, cell storage and derivative helpers are private. The method
  inherits `cuda::Driver<gpw::Phase>` and mounts directly on the parent's
  Backend, Frame and stream. Three original scientific paths share one SCF
  iteration loop; the fused DIIS/eigen/density primitive retains its numerical
  order. Derivatives use that solve's state. Live-bound moves, requests,
  standalone and nested stage/criterion lifecycles, exception/retry and
  geometry reuse pass in the [native envelope](../../tools/diagnostics/gpw-driver/results/2026-10-27/gh200/validation.json).
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

- Forces: device grid Hellmann–Feynman + **host AO Pulay** residual. Frame
  `stress(frame)` performs one device SCF and passes that result to the existing
  host fixed-density stress assembly, with no CPU SCF. `stress_strain_h` and
  `stress_axes` match the CPU knobs. It returns intensive stress in eV/Å³ and
  overwrites the extensive `kStress = sigma * frame.box.volume()` in eV;
  `kEnergy` is the energy from the same device solve, and forces are preserved.
  The algorithm remains a central strain difference with diagonal strains
  only; shear is zero, and no complete device-residency claim follows.
- Evidence class is CPU-twin (◐), not CP2K; promoting to ✓ needs independent
  CP2K pins.

## Open depth (do not soft-land)

1. `device-cold-start` — densify still on host.
2. Multi-k χ / orth / Vnl / H0 residency, payload and cold-start.
3. Device-native AO Pulay / force handoff; device stress kernels.
4. Production multi-k / MG / UKS × PBE joints; Si MG F/σ.
The nested Driver migration is complete within its stated native envelope;
the residency work above remains open and cannot inherit that acceptance.

## Forbidden

A public CUDA GPW API that differs from CPU `GpwDft` (Config bag, free-loop
solver type, foothold-as-product); promoting `KsChainDevice` to user surface;
softened or self-golden CUDA↔CPU pins; claiming green without GH200 evidence.
