# Spec — gpw-bands-metallic

## Summary

GPW band structure under **metallic FD/MP occupation smearing**: SCF with
smear → freeze \(V_{\mathrm{eff}}\) → NSCF path (or equivalent product path)
for a metallic GTH-GPW system.

## Domain basis

- Insulating RKS bands: `reg_gpw-bands-si` ✓
- Metallic smear energy: `reg_gpw-metallic-smearing` ✓
- Metallic DOS: `reg_gpw-dos-metallic` ✓
- Residual: no dedicated bands pin under smear (Fermi level / metallic path).

## Design

1. `cpu::gpw::Bands` (or metallic mode) runs multi-k SCF with `Smearing` set,
   freezes veff, NSCF along path; report \(E_F\) and path eigenvalues.
2. Twin: CP2K 2026.1 `SMEAR` + `&PRINT &BAND_STRUCTURE` on GTH-GPW metal
   (e.g. BCC Na class — matched GPW, not NC-UPF PW).
3. Dual-gate: free-energy / \(E_F\) character vs CP2K; fair wall on twin job.

## Files (expected)

| Path | Role |
| --- | --- |
| bands + smear integration | product path |
| `regressions/gpw/gpw-bands-metallic/` | reg + REFERENCE |
| `tools/cp2k_oracle/inputs/gpw-bands-metallic/` | CP2K twin |

## Tasks

1. Implement metallic bands path.
2. Offline CP2K twin + pins.
3. Regression dual-gate + fair wall ≤ 10.

## Out of scope

- UKS metallic bands combo; PW NC-UPF bands.
