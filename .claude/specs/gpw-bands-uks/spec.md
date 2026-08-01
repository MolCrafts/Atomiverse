# Spec — gpw-bands-uks

## Summary

Collinear **UKS** band structure on GPW: multi-k SCF → freeze \(V_{\mathrm{eff}}^\sigma\)
→ NSCF along a high-symmetry path (same shape as `cpu::gpw::Bands` RKS product
`gpw-bands-si`).

## Domain basis

- RKS GPW bands: `reg_gpw-bands-si` ✓
- Multi-k UKS SCF: `reg_gpw-kpoints-uks` ✓
- Residual: bands post-process is RKS-only (or unpinned under UKS).

## Design

1. Extend `cpu::gpw::Bands` (or dual-spin twin) to accept UKS `GpwDft`
   (`n_alpha`/`n_beta`): SCF once, freeze spin-resolved veff, NSCF path
   eigenvalues per spin.
2. Twin: CP2K 2026.1 Quickstep UKS + `&PRINT &BAND_STRUCTURE` on GTH-GPW Si
   (matched GPW representation — not PW NC-UPF).
3. Dual-gate: total energy vs CP2K SCF; path character (gap / spin channels);
   fair wall on SCF+bands twin.

## Files (expected)

| Path | Role |
| --- | --- |
| `include/atomiverse/cpu/gpw/` bands surface | UKS path |
| `regressions/gpw/gpw-bands-uks/` | reg + REFERENCE |
| `tools/cp2k_oracle/inputs/gpw-bands-uks/` | CP2K twin |

## Tasks

1. Implement UKS bands path on product surface.
2. Offline CP2K twin + REFERENCE pins.
3. Regression dual-gate + fair wall ≤ 10.

## Out of scope

- Metallic smear bands (`gpw-bands-metallic`); DOS (`gpw-dos-uks` already landed).
