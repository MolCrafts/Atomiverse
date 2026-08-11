# cuda-gpw-product-collocate-pbc

method: **cuda** | track: product depth P0 | parent: `dev` @ product-si + residency

## Problem

Continuous device product collocate (`product_collocate.cu`) evaluates
`shape(r) = w · (r−ra)^la (r−rb)^lb · exp(−p|r−pc|²)` with **raw**
displacements. On production Si (edge / face product centers) this under-counts
vs CPU collint (sample charge ratio ~0.47–0.54). That defect forced the SG RKS
product mid-SCF path onto a **host CompactBasisGrid** twin + mid-path **P D2H**
(`gpw_solver.cu`) — iron-law debt after `cuda-gpw-product-si-gate`.

## Lever

| File | Function | Metric |
| --- | --- | --- |
| `src/cuda/gpw/internal/product_collocate.cu` | `collocate_products_sphere_kernel` | sphere-window unwrapped r + wrap store; Si Q≈16; edge vs collint |
| `src/cuda/gpw/internal/product_integrate.cu` | `integrate_products_sphere_kernel` | integrate twin of sphere collocate |
| `src/cuda/gpw/gpw_solver.cu` | `build_energy_device` SG branch | restore device pack→`KsChainDevice`→scatter; **zero** mid-SCF scientific DH of P/ρ/V |

Skills: `choose-launch-configuration` (128 product-parallel / 256 full-grid),
`optimize-global-memory-access` (window locality + atomicAdd),
boundary wrap (unwrapped eval / wrap store), `av-perf` residency (drop host cbg mid-SCF).

## Acceptance

1. Unit: edge-centered s–s product — device ∫ρ and max\|Δρ\| vs collint within stated tol; interior centers still match existing footholds.
2. SG RKS mid-SCF: no `DeviceToHost`/`HostToDevice` of scientific matrices (P, ρ, V_AO, H) — scalar residual/energy OK; cbg retained **only** for force free-fn / boundary.
3. GH200: `reg_cuda-gpw-product-si-gate` holds (\|dE\| ≤ 5e-5 Ha) + `reg_cuda-gpw-product-collocate-pbc` PASS.
4. **perf_exempt** (correctness / residency; not fair-wall).

## Out of scope

H0/KB (landed), Si MG energy path (child, unblocked by this), MG F/σ, multi-k.
