# Reference — GPW PBE + rVV10 self-consistent nonlocal potential + forces

## Method

**Self-consistent** nonlocal rVV10 correlation on Γ-point **GPW PBE** (periodic
water dimer, same geometry as `gpw-rvv10-ice`). Product path:

1. Each SCF iteration: \(V_{\mathrm{nl}}[\rho]=\delta E_c^{\mathrm{nl}}/\delta\rho\)
   (Soler / CP2K `qs_dispersion_nonloc`) added to KS \(V_{\mathrm{eff}}\).
2. Final \(E_{\mathrm{disp}}=E_c^{\mathrm{nl}}\) on the SC density →
   `GpwResult::e_disp` [Hartree].
3. Analytic forces reuse \(V_{\mathrm{nl}}\) inside `assemble_gpw_forces` veff.

Public API: `cpu::gpw::GpwDft::rvv10 = true` (also `vdw_df` / `vdw_df2` for
DRSLL / LMKLL). Same exclusivity rules as the ice pins.

CP2K twin: `&VDW_POTENTIAL` / `POTENTIAL_TYPE NON_LOCAL` /
`&NON_LOCAL TYPE RVV10` PARAMETERS 6.3 0.0093 (self-consistent by default).

## Reference values (hard gate)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp (gate)** | **CP2K 2026.1** TYPE RVV10 SC | **+6.116467960721×10⁻² Ha** |
| E_disp (ATV SC product) | this regression | +6.11479080989958×10⁻² Ha |
| E_disp (post-SCF ATV, historical) | `gpw-rvv10-ice` | +6.115349220971×10⁻² Ha |
| Force FD | ATV analytic vs central FD \(h=5\times10^{-4}\) Bohr on O0 \(x\) | \|ΔF\| ≈ **1×10⁻⁶ Ha/Bohr** (gate 5×10⁻⁴) |

Regression tolerance on E_disp: **1×10⁻⁴ Ha** vs CP2K SC pin
(Δ(ATV SC − CP2K) ≈ −1.7×10⁻⁵ Ha; single-grid vs CP2K multigrid residual).

### Offline notes

Input twin: `tools/cp2k_oracle/inputs/gpw-vdw-scf-potential/water_dimer_rvv10_force.inp`
(and `gpw-rvv10-ice` energy twin). Host: Arrhenius `work/cp2k/install` CP2K 2026.1.

```text
Dispersion energy = 0.06116467960721 Ha
Total energy      = -34.27712583714492 Ha
wall ≈ 121 s (OMP=8)
```

## Geometry

Same water dimer as `gpw-rvv10-ice` (12 Å cubic, DZVP-MOLOPT-SR-GTH + GTH-PBE,
CUTOFF 200 Ry, PERIODIC XYZ).

## Perf gate

**perf_exempt = true**: SC RPS costs \(O(N_{\mathrm{SCF}}\times n_\alpha)\) FFTs;
primary deliverable is potential/force correctness (same policy as ice E_disp pins).
