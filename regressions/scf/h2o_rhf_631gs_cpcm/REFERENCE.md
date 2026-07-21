# H₂O RHF/6-31G* C-PCM electrostatic free energy

## System

- Geometry (Å): O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) — same as other
  H₂O regressions in this tree.
- Method: RHF / 6-31G* (`Pople631Gs`).
- Solvent: water-like dielectric **ε = 78.4**, cavity = **Bondi × 1.2**,
  dielectric scaling **x = 0** (C-PCM, Barone–Cossi), Lebedev **302** points
  per sphere with hard bury-delete of buried tesserae.
- Quantity: electrostatic free energy **ΔG_el = ½ Σ_k q_k V_k** (Hartree →
  kcal/mol with 627.5094740631). Non-electrostatic terms (cavitation,
  dispersion, repulsion) are **not** included.

## External pin (PySCF 2.14.0)

```python
from pyscf import gto, scf
mol = gto.M(
    atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    basis="6-31g*", unit="Angstrom", verbose=0)
mf = scf.RHF(mol).PCM()
mf.with_solvent.method = "C-PCM"
mf.with_solvent.eps = 78.4
mf.with_solvent.vdw_scale = 1.2
mf.with_solvent.lebedev_order = 29   # 302 grids
mf.conv_tol = 1e-10
mf.kernel()
print(mf.with_solvent.e)             # Hartree
```

| Quantity | Value |
|---|---|
| PySCF version | **2.14.0** |
| `mf.with_solvent.e` (ΔG_el) | **−0.01268465021284 Ha** |
| ΔG_el [kcal/mol] | **−7.95973818** |
| E_gas (RHF/6-31G*) | −76.00910803237647 Ha |
| E_C-PCM total | −76.02066073847446 Ha |

Notes on model differences (honest):

- Both use **modified Bondi H = 1.1 Å**, scale 1.2, ε = 78.4, x = 0 (C-PCM),
  Lebedev 302, and the Lange–Herbert Gaussian-smeared S matrix (erf(ξr)/r).
- PySCF PCM additionally applies **SWIG** surface switching
  (Lange & Herbert, JCP 133, 244111 (2010)); Atomiverse uses **hard
  bury-delete** (point inside any other sphere is discarded). The cavity
  surface is therefore discontinuous in nuclear coordinates — intentional.
- AO potentials in Atomiverse use the point-charge nuclear-attraction
  primitive; PySCF uses Gaussian-charge `int3c2e`. For H₂O the residual
  is well inside the 0.1 kcal/mol gate.

Measured Atomiverse ΔG_el on this geometry: **−8.018 kcal/mol**
(|Δ − pin| ≈ 0.059 kcal/mol).

## Acceptance

- `|ΔG_el(Atomiverse) − pin| ≤ 0.1 kcal/mol`
- Pin lies in **[−8, −6] kcal/mol**
- Total C-PCM energy strictly below gas phase; SCF converges ≤ 60 s
