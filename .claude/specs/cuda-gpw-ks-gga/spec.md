# cuda-gpw-ks-gga

## Lever

Extend the landed Γ-point composed KS stage (`KsChainDevice`) from Padé LDA
to **closed-shell PBE-GGA**:

1. Spectral ∇ρ on the full r2c half-complex grid (CPU `gradient` conventions:
   unnormalized r2c, `i G_d` with Nyquist zeroed, c2r / N).
2. Pointwise closed-shell PBE via reused `cuda::pw::internal::xc_pbe_grid`
   (local `v_ρ`, `ε_xc`, flux written over gradients).
3. Spectral −∇·flux (CPU `divergence` conventions) → `V_xc = v_ρ − div`.
4. Existing Poisson + V_eff assemble + product integrate bookends unchanged.

**Stage unit only** — not public `GpwDft::run`, no multigrid, no spin.

| Item | Value |
| --- | --- |
| Track | method=cuda |
| Files | `ks_chain.{cuh,cu}`, new `spectral_fft.{cuh,cu}`, CMake; unit + reg |
| Twin | CPU `ks_grid_build_rks(..., XcKind::PBE)` on small Γ product grid |
| Metric | max\|ΔV_eff\|, \|Δe_H\|, \|Δe_xc\|, max\|Δacc\| (Hartree) |
| Acceptance | GH200 unit + dual-gate green; stated tols; device-only kernels |

## Out of scope

Multigrid chain, UKS GGA, public method product, hybrid/meta-GGA, forces/stress.
