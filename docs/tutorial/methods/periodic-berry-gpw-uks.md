# Collinear UKS Berry-phase polarization on GPW — bulk Si (KSV)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-berry-gpw-uks` |
| Domain | response / GPW |
| Oracle | CP2K 2026.1 Quickstep `&MOMENTS PERIODIC` + `LSD` (Γ) |
| Ratio | \(T_\mathrm{atv}/T_\mathrm{cp2k} \approx 0.214\) |

## Summary

King–Smith–Vanderbilt formal polarization on collinear UKS GPW
(`cpu::gpw::BerryPolarization` dual-channel string phases). Public
members only; no `*Config`. Pins: centrosymmetric half-quantum lattice
+ elemental Born ASR \(Z^*=0\).

## Method

Per-spin occupied link determinants on string meshes; electronic
polarization \(p^\mathrm{el}=(\varphi_\uparrow+\varphi_\downarrow)/2\pi\),
wrap quantum 1. Insulator only (metallic smear rejected).

## Implementation

```cpp
// frame: Si positions [Å], Z = 14 in kAtom_type, orthorhombic box [Å]
GpwDft m{{Si}, 300.0, XcKind::PADE}; m.n_alpha = 8; m.n_beta = 8; m.n_occ = 0; m.build(frame);
BerryPolarization berry; // k_perp, k_string, directions
auto r = berry.compute(m, frame);
```

Run: `cmake --build build --target run_reg_periodic-berry-gpw-uks`.

## Validation

See pin notes. Measured \(Z^*=+0.0026\); ideal \(p=(0,0,0)\).

## Performance

Fair wall: one ATV string-mesh direction (21.09 s) vs CP2K ENERGY+MOMENTS
(98.68 s), OMP=8, 1 MPI. `within_10x=true`.

## Limitations

CP2K k-point PERIODIC moments are closed-shell only (2026.1); twin is Γ
UKS Berry-phase moments. Non-orthorhombic / metallic / DFPT Born out of
scope.

## References

- case `periodic-berry-gpw-uks` pin
- King-Smith & Vanderbilt, *Phys. Rev. B* **47**, 1651 (1993)
- Spaldin, *J. Solid State Chem.* **195**, 2 (2012)
