# H₂O RHF/6-31G* IEF-PCM electrostatic free energy

## System

- Geometry (Å): O (0, 0, 0.1173), H (0, ±0.7572, −0.4692) — same as other
  H₂O regressions in this tree.
- Method: RHF / 6-31G* (`Pople631Gs`, Cartesian d shells).
- Solvent: water-like dielectric **ε = 78.4**, cavity = **Bondi × 1.2**,
  model = **IEF-PCM** (Cancès–Mennucci–Tomasi; PySCF `method='IEF-PCM'`),
  Lebedev **302** points per sphere. Atomiverse uses hard bury-delete of
  buried tesserae (PySCF default is SWIG surface switching).
- Quantity: electrostatic free energy **ΔG_el = ½ Σ_k q_k V_k** (Hartree).
  Non-electrostatic terms (cavitation, dispersion, repulsion) are **not**
  included. Observable `e_solv` ≡ `e_solvent()` ≡ ΔG_el.

## External pin (PySCF 2.14.0)

Offline oracle (never imported by ctest):

```bash
source tools/setup_pyscf_env.sh
python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis '6-31g*' \
  --cart --pcm --pcm-method 'IEF-PCM' --pcm-eps 78.4 --pcm-scale 1.2 \
  --pcm-lebedev-order 29 --conv-tol 1e-10
```

Equivalent script:

```python
from pyscf import gto, scf
mol = gto.M(
    atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    basis="6-31g*", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol).PCM()
mf.with_solvent.method = "IEF-PCM"
mf.with_solvent.eps = 78.4
mf.with_solvent.vdw_scale = 1.2
mf.with_solvent.lebedev_order = 29   # 302 grids
mf.conv_tol = 1e-10
mf.kernel()
print(mf.with_solvent.e)             # Hartree ΔG_el
```

| Quantity | Value |
|---|---|
| PySCF version | **2.14.0** |
| `mf.with_solvent.e` (ΔG_el / e_solv) | **−0.01265803490853 Ha** |
| ΔG_el [kcal/mol] | **−7.94303683** |
| E_gas (RHF/6-31G* cart) | −76.01050498825606 Ha |
| E_IEF-PCM total | −76.02202516714988 Ha |

Notes on model differences (honest):

- Both use **modified Bondi H = 1.1 Å**, scale 1.2, ε = 78.4,
  f = (ε−1)/(ε+1), Lebedev 302, and the Lange–Herbert Gaussian-smeared S/D
  matrices (erf(ξr)/r).
- PySCF PCM additionally applies **SWIG** surface switching
  (Lange & Herbert, JCP 133, 244111 (2010)); Atomiverse uses **hard
  bury-delete** (point inside any other sphere is discarded). The cavity
  surface is therefore discontinuous in nuclear coordinates — intentional.
- AO potentials in Atomiverse use the point-charge nuclear-attraction
  primitive; PySCF uses Gaussian-charge `int3c2e`. Residual vs pin is gated
  at **1e-4 Ha** (~0.06 kcal/mol), consistent with the C-PCM twin
  (`h2o_rhf_631gs_cpcm`).

## Acceptance

- `|e_solv(Atomiverse) − pin| ≤ 1e-4 Ha` (and pin is the literature oracle)
- `e_solv < 0` (stabilizing electrostatic free energy)
- Total IEF-PCM energy strictly below gas phase; SCF converges ≤ 60 s

Public API:

```cpp
cpu::scf::RHF sol;
sol.basis = cpu::scf::BasisSet::Pople631Gs;
sol.solvent_eps = 78.4;
sol.solvent_radii_scale = 1.2;
sol.solvent_model = cpu::scf::SolventModel::IEFPCM;
sol.run(frame);
double e_solv = sol.e_solvent(); // [Hartree]
```
