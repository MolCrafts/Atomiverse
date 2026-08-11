# CUDA GPW public-shape freeze

Binding contract for `cuda-gpw-full-product`. Landed by
`arch-cuda-gpw-public-shape-freeze` (method=arch-doc; merge `0147fe02` /
docs `aea603bb`). Current understanding
only — not a history log.

## Lever

`MeanFieldMethod` + `cpu/scf/stages/` (typed SCF Driver stages;
`run(frame)` / `forces(frame)`) and multi-path DIIS `GpwDft :
cpu::Driver<gpw::Phase>` (`ks_build → mix → diagonalize → density → check`).
CUDA classical twins: `cuda::scf::RHF`, `cuda::pw::PW`. Full product **equals**
that public shape; it does **not** invent a second SCF surface from stage
footholds.

## Status today

| Layer | What exists |
| --- | --- |
| CPU method | `cpu::gpw::GpwDft` — multi-path **DIIS** on Driver stages; **OT** free-loop residual (same capability follow-on, not a block) |
| CUDA stages | `cuda/gpw/internal/`: Poisson, XC LDA, product collocate/integrate, composed `KsChainDevice` (LDA+PBE), `KsMultiGridChainDevice` |
| CUDA public method | **product landed** (`cuda::gpw::GpwDft` Γ RKS DIIS PADE/PBE + **Γ UKS DIIS PADE** + **multigrid energy** + **water dual-gate** + **multi-k RKS DIIS energy** H₂ foothold; device RKS DIIS/eigen + product collocate/KS; UKS host dual GpwDiis; multi-k host χ^k residual; free-fn forces hybrid) — depth residual on queue |

## Frozen public shape (CUDA product must match)

When `cuda-gpw-full-product` lands, the user-facing method is the CPU twin:

```text
// Construction → members → build → run → last_result
// Forces/stress free functions or Frame evaluator — same idea as CPU
cuda::gpw::GpwDft method;          // name may match; namespace cuda::gpw
method.cutoff_ry = …;              // members, not *Config
method.build(/* reference system */);
auto& res = method.run(/* geometry */);
const auto& last = method.last_result();
// forces: free function or GPWEvaluator-class Frame path (not a second solver)
```

### Required (first method product)

- Concrete method type under `ATV::cuda::gpw` (not freestanding `SCFSolver`)
- Parameters as public members / ctor args — **no** public `*Config` bags
- `build` fail-fast; re-entrant `run`; `last_result`
- Forces path present for the shipped capability slice (analytic free function
  and/or Frame evaluator writing `kAtom_f*` / energy meta) — same shape family
  as CPU `gpw_forces_analytic` / `GPWEvaluator`
- Stage kernels stay under `cuda/gpw/internal/`; product **composes** them
- No `make_*` / `build_*` factories returning constructed methods
- Users never define Context or hand-drive Component lifecycle

### Optional depth (not required for first product)

Post-SCF (`mp2` / RPA / G0W0), OT minimizer, multi-k UKS, vdW stack, etc. Ship
when CPU twin is green and CUDA↔CPU tols are stated. **Capability subset is
allowed; public-shape fork is not.**

### Explicit non-product (stay internal)

`KsChainDevice`, `KsMultiGridChainDevice`, `PoissonFftDevice`,
`DeviceProduct` collocate/integrate, spectral MG transfer, XC grid kernels.
These are stage/oracle units (`tests/cuda/gpw/*`, `reg_cuda-gpw-ks-*`). They
must not become the public method API.

## OT residual does not block

CPU OT (`use_ot`) remains free-loop residual after multi-path DIIS Driver.
That residual does **not** block this freeze and does **not** force OT onto
the first CUDA product slice. DIIS Γ (and later multi-path DIIS) is the
default product spine; OT is same-capability follow-on on both backends when
staffed.

## Unlock condition for `cuda-gpw-full-product`

| Gate | State on `dev` @ `9c331b9d` |
| --- | --- |
| Multi-path DIIS CPU Driver | **green** (`arch-gpw-driver-paths`) |
| Public-shape freeze | **green** (this note; merge `0147fe02`) |
| Stage footholds composed | **green** (ks-chain / GGA / MG units) |
| First method product | **landed** — feat `31d24207`; GH200 **1120817** dual-gate (PADE); device collocate **1125709**; MG energy **1126433**; Γ UKS **1126430** + O₂ **1127173**; device forces **1127167**; PBE H₂ **1127166**; stress **1128195**; non-default stream **1128196**; device RKS DIIS/eigen **1129719**; water dual-gate **1130195**; multi-k energy **1130141** (H₂ host residual); residual (multi-k device spine + Si/water multi-k gates, MG/UKS F/σ + joints, force-stream / AO-Pulay, device STVnl/weight-pack/uks-diis) stays on es-cpu-cuda queue |

`cuda-gpw-full-product` is no longer blocked on “needs architecture freeze”.
It remains a product implementation slug (compose internals → public method),
subject to ES CUDA open-gate rules in CLAUDE.md.

## Harness

`regressions/architecture/arch-cuda-gpw-public-shape-freeze/` — shape-only
gate on the CPU public contract the CUDA product must mirror. No multi-iter
SCF; no tree-scan style gates.

## Forbidden

- Public CUDA GPW API that differs from CPU `GpwDft` shape (e.g. Config bag,
  free-loop solver type, foothold-as-product)
- Staffing full product by promoting `KsChainDevice` to user surface
- Softening pins / self-golden CUDA↔CPU
- Claiming product green without GH200 evidence
