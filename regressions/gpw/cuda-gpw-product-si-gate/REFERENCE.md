# `cuda-gpw-product-si-gate`

## What

Production **Si-class Γ RKS PADE single-grid** (`ngrids=1`) energy dual-gate on
public `cuda::gpw::GpwDft` vs same-settings CPU. Isolates H0/STVpp (GTH
nonlocal KB) after H₂/water product footholds hide Si nonlocal production.

## Twin

| | |
| --- | --- |
| System | diamond Si bct 4-atom, a = 5.43 Å |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| Settings | Γ, CUTOFF 40 Ry, ngrids=1, n_occ=8, PADE, DIIS |
| Oracle | **same-settings CPU** |

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| Both SCF converged | required |

## Lever

1. CUDA Γ Vnl was per-pair `gth_nonlocal` with ket-only lattice images (wrong
   for periodic Si nonlocal). Fixed to shared `add_nonlocal_kb_k` at Γ
   (same KB form as CPU `add_nonlocal_kb` / multi-k STVnl host path).
2. Continuous device product collocate has **no collint PBC window wrap** —
   H₂/water large-cell hid it; Si MOLOPT tight cell under-counts density
   (GH200 **1166099**/**1166106**: after KB fix, CUDA still non-convergent
   E≈−5.18 vs CPU −15.15). Mid-SCF uses **host collint** ρ+KS+integrate twin
   + device H=H0+V / DIIS. Device continuous restore → `product-collocate-pbc`.

## Perf

**perf_exempt** — correctness dual-gate / host collint residual (not fair-wall).

## Status

GH200 dual-gate pending after host-collint hybrid + KB fix.
