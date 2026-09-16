# `cuda-gpw-product-h2o-stress`

## What

Production water Γ RKS PADE **cell stress** dual-gate. Residual after
`product-h2o-gate` (E+F) and H₂ `cuda-gpw-stress`. Host fixed-\(P\) assembly
at the CUDA density payload (no new kernel).

## Twin

Same-settings CPU `cpu::gpw::GpwDft` / `gpw_stress` as
`cuda-gpw-product-h2o-gate` (H₂O, O SZV + H DZV, 12 Bohr, 100 Ry, Padé,
`n_occ=4`). Oracle is the CPU twin, not CP2K (CUDA rejects MT).

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| max‖Δσ‖ (xx,yy,zz) | **5×10⁻⁵ Ha/Bohr³** |
| Off-diagonal | **1×10⁻⁸** |

**Claim: ◐** — CPU-twin dual-gate; not an independent CP2K pin (CUDA
rejects Martyna–Tuckerman). **perf_exempt** — small-molecule correctness-only.
