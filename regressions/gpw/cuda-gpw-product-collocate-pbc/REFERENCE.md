# cuda-gpw-product-collocate-pbc

## What

Continuous device product collocate/integrate under **PBC min-image**, and
restored single-grid RKS device KS mid-path (no host CompactBasisGrid twin).

## Oracle

- **Collocate charge:** CPU `collint_collocate_product` on the same edge-centered
  product (same-settings continuous/collint twin).
- **Energy dual-gate:** same-settings CPU `GpwDft` on H₂ Γ RKS PADE ngrids=1
  (proves SG device product path still matches CPU class energy).

## Provenance

Offline pins only. No live CP2K. Si production dual-gate remains
`reg_cuda-gpw-product-si-gate` (must hold on GH200 with this change).

## Tolerances

| Quantity | Tol |
| --- | ---: |
| edge charge \|Q_dev − Q_collint\| / \|Q\| | 5e-4 relative |
| H₂ \|E_cuda − E_cpu\| | 5e-6 Ha (same class as full-product) |

**perf_exempt** — correctness / iron-law residency, not fair-wall.
