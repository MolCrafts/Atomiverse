<!-- mol:note:topic:cuda-gpw-product-inventory -->
# CUDA GPW product inventory (current understanding)

What is **on `dev` now** for `cuda::gpw::GpwDft` — architecture check only.
Not a history log. Binding contracts: [cuda-gpw-public-shape.md](cuda-gpw-public-shape.md),
[cuda-only-no-dh.md](cuda-only-no-dh.md).

## Public shape — **OK**

| Rule | Status |
| --- | --- |
| Concrete method `ATV::cuda::gpw::GpwDft` | yes |
| Members / no `*Config` / no `make_*` | yes |
| `build` → `run` → `last_result` | yes |
| Forces/stress free-fn family (CPU twin idea) | yes (`gpw_forces_*`, stress free-fn) |
| Stages under `cuda/gpw/internal/` only | yes |
| Same `GpwAtom` / `GpwResult` / `XcKind` as CPU | yes |
| Fail-closed unsupported (OT/MT/…) | yes |

**Architecture residual (not a fork):** CUDA method is still a free-loop
composer inside `run()`, not `cuda::Driver<gpw::Phase>` nest — same class of
debt as `cuda::scf::RHF` vs nested DriverNode. **Do not invent a second public
surface to “fix” it.**

## Device composition (mid-SCF)

```text
setup (host once)
  S/T/Vnl (Γ: KB via add_nonlocal_kb_k@κ=0)  →  H0, orth
  product list / monom tasks / tables H2D
  multi-k: χ^k + T_k/Vnl_k/H0_k H2D once

mid-SCF (device, non-default stream)
  P resident → weight pack
  ρ collocate (SG PBC sphere-window | MG monom multi-level)
  Poisson + XC → V_eff
  integrate → V_AO scatter
  H = H0 + V  (device)
  DIIS residual (device scalar) → eigen → P  (AoDiis / KpointDiis)

boundary
  final eigen_from_device → C/ε (+ H) D2H once
  optional V_eff/V_es D2H for force free-fn
```

| Path | Mid-SCF science on device? | Dual-gate class |
| --- | --- | --- |
| Γ RKS single-grid | **yes** (pack→KS→H→DIIS) | H₂ / water / Si ngrids=1 |
| Γ RKS multigrid | **yes** monom default (`ATV_GPW_CUDA_HOST_MG=1` debug only) | H₂ MG + Si ngrids=4 monom |
| Γ UKS single-grid | **yes** dual pack + dual H/DIIS + dual payload | O₂-class payload |
| Multi-k RKS | mid-SCF yes; setup χ/STVnl pieces hybrid | H₂ k-mesh foothold |

## Internals map

| Component | Role |
| --- | --- |
| `ProductWeightPackDevice` | P→product weights; acc→V_AO scatter |
| `product_collocate` / `product_integrate` | monom sphere-window PBC |
| `KsChainDevice` / `KsMultiGridChainDevice` | ρ→V_eff (+ prolong/restrict) |
| `AoDiisEigenDevice` | Γ RKS/UKS DIIS + eigen + density + final payload |
| `KpointStvnlDevice` / collocate / integrate / DIIS | multi-k spine |
| free-fn forces | device grid HF + **host AO Pulay** residual |

## Evidence spine (high-signal GH200)

| Capability | Job / merge |
| --- | --- |
| residency P0 | **1158949** / `5a5076d0` |
| product-si (Si SG) | **1167099** / `a158848f` |
| collocate-pbc + SG device KS | **1173564** / `cfb246d1` |
| Si MG energy | **1180883** / `88d8ec59` |
| device monom MG default | **1186912** / `3730ada7` |
| RKS device payload | **1182615** / `d476ee7a` |
| UKS device payload | **1186414** / `f2be952a` |
| Si / H2 / H2O product dual-gate | **1331963** (`GPU_JOB_OK`, Si \|dE\|=3.529e-05 Ha) |
| full-product H2 energy+F | **1332946** `GPU_JOB_OK` (ks_chain + product + reg; \|dE\|=1.205e-08, max\|dF\|=3.230e-06). Prior 1331962 unit-only V_eff fail, now 1e-6·peak. |

## Open depth (do not soft-land)

1. `multigrid-si-gate` — production dual-gate after monom on base  
2. `device-cold-start` — densify still host  
3. Multi-k: χ / orth / vnl / h0 residency + payload/cold-start  
4. Force AO-Pulay / force-handoff; stress device  
5. Production multi-k/MG/UKS×PBE + MG F/σ matrix  
6. Nested `cuda::Driver<gpw::Phase>` (shape-preserving fold only)

## Verdict

- **Public architecture:** correct and freeze-compliant.  
- **Iron-law mid-path (Γ product RKS/UKS + MG monom):** closed for claimed paths.  
- **Optimization next:** multi-k setup residency and free-fn AO Pulay (real PCIe
  wins on large cells), not more SG dual-gate soft-tols.  
- **CPU wall track:** exhausted pin-safe staff (`cpu_pending=0`); MG MAIN 2.33×
  remains blocked pin-unsafe.
