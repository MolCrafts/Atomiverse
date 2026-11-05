# GPW Berry-phase polarization

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

King–Smith–Vanderbilt (KSV) formal polarization of a periodic insulator on the
CPU GPW method: string meshes of k-points along each reciprocal lattice vector,
occupied link determinants between neighbouring k-points, and the electronic
polarization from the string-averaged Berry phase. `cpu::gpw::BerryPolarization`
(public members `k_perp`, `k_string`, `directions`) takes a built
`cpu::gpw::GpwDft` and the Frame and returns phases, electronic / ionic / total
fractional polarization and the dipole; no `*Config` bag, no factory.

The case runs bulk diamond Si in two sections: `periodic-berry-gpw`
(closed-shell RKS, `n_occ = 8`) and `periodic-berry-gpw-uks` (collinear UKS,
`n_alpha = n_beta = 8`). Both are gated on the same two symmetry-exact
literature pins; the CP2K twin supplies the fair wall. Only the UKS variant has
a section on this page; the RKS record lives in the case `REFERENCE.md`.

Case: `ctest -R '^gpw/berry-phase$'` — sources and oracle record in [`regressions/gpw/berry-phase/`](../../../regressions/gpw/berry-phase/). One section of the case per `##` heading below: `atv_regressions --case gpw/berry-phase --section <name>`.

## Common setup

- **System:** diamond Si, 4-atom body-centred tetragonal orthorhombic cell,
  \(a = 5.43\) Å, DZVP-MOLOPT-GTH / GTH-PBE-q4, `XcKind::PADE`, 80 Ry density
  cutoff, SCF tol \(10^{-7}\) Ha; strings `k_perp = 1`, `k_string = 4`.
- **Pins:** (1) centrosymmetric crystal ⇒ every ideal-Si \(p_i\) on the
  half-quantum lattice; (2) elemental crystal ⇒ Born effective charge
  \(Z^* = 0\) under a sublattice displacement (acoustic sum rule).
- **Oracle:** CP2K 2026.1 Quickstep `&MOMENTS PERIODIC`; matched geometry,
  GTH-PADE, MOLOPT, PADE, 80 Ry. The fair wall compares one ATV string-mesh
  SCF + Berry direction with one CP2K ENERGY + MOMENTS job (1 MPI, OMP=8).
- **Scope:** insulators only (metallic smearing is rejected); orthorhombic
  cells. Non-orthorhombic cells and DFPT Born charges are out of scope.

## References

- King-Smith & Vanderbilt, *Phys. Rev. B* **47**, 1651 (1993)
- Spaldin, *J. Solid State Chem.* **195**, 2 (2012)

## periodic-berry-gpw-uks — Collinear UKS Berry-phase polarization, bulk Si {#periodic-berry-gpw-uks}

### Method

Per-spin occupied link determinants on the string meshes (dual-channel string
phases); electronic polarization
\(p^\mathrm{el}=(\varphi_\uparrow+\varphi_\downarrow)/2\pi\), wrap quantum 1.

```cpp
// frame: Si positions [Å], Z = 14 in kAtom_atomic_number, orthorhombic box [Å]
GpwDft m{{Si}, /*cutoff_ry=*/80.0, XcKind::PADE};
m.n_alpha = 8; m.n_beta = 8; m.n_occ = 0;
m.build(frame);
BerryPolarization berry;   // k_perp, k_string, directions
auto r = berry.compute(m, frame);
```

### Validation

Measured \(Z^*=+0.0026\); ideal \(p=(0,0,0)\).

### Performance

Fair wall: one ATV string-mesh direction (21.09 s) vs CP2K ENERGY+MOMENTS
(98.68 s), OMP=8, 1 MPI: \(T_\mathrm{atv}/T_\mathrm{cp2k} \approx 0.214\),
`within_10x=true`.

### Limitations

CP2K k-point PERIODIC moments are closed-shell only (2026.1), so the twin is
Γ UKS Berry-phase moments (`LSD`).
