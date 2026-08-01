# Spec — gpw-uks-metallic-stress

## Summary

Multi-k **UKS** free-energy analytic cell stress under FD/MP metallic
smearing (collinear dual \(P_k^\sigma/W_k^\sigma\)), sibling of landed
`gpw-metallic-stress` (RKS) and `gpw-uks-metallic-forces`.

## Domain basis

- RKS metallic stress: `reg_gpw-metallic-stress` ✓
- UKS metallic forces: `reg_gpw-uks-metallic-forces` ✓
- Multi-k UKS stress (integer occ): `reg_gpw-kpoint-uks-stress` ✓
- Residual: free-energy UKS stress under fractional occupations unpinned.

## Design

1. Stress assembly rebuilds spin-resolved \(P_k^\sigma\), \(W_k^\sigma\) from
   fractional `f_occ_k` under smear (same payload rules as UKS metallic forces).
2. Dual-gate analytic vs multi-SCF FD of free energy; offline CP2K 2026.1 UKS
   `STRESS_TENSOR NUMERICAL` + `SMEAR` twin (matched GTH-GPW).
3. Fair wall on ENERGY (or stress) twin SCF job ≤ 10×.

## Files (expected)

| Path | Role |
| --- | --- |
| `gpw_stress.cpp` / UKS smear payload | product |
| `regressions/gpw/gpw-uks-metallic-stress/` | reg + REFERENCE |
| `tools/cp2k_oracle/inputs/gpw-uks-metallic-stress/` | CP2K twin |

## Tasks

1. Wire UKS smear stress if missing; dual-gate unit coverage if extractable.
2. Offline CP2K twin + REFERENCE.
3. Regression + fair wall ≤ 10.

## Out of scope

- RKS metallic stress (done); variable-cell CELL_OPT under smear.
