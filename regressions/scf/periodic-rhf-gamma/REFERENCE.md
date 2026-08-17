# `periodic-rhf-gamma`

H₂ / STO-3G Cartesian **Γ-point periodic RHF** in a 20 Å cubic cell.

## Geometry

| Atom | x / Å | y / Å | z / Å |
| ---- | ----- | ----- | ----- |
| H    | 0     | 0     | 0     |
| H    | 0     | 0     | 0.74  |

`Frame.box = cubic(20.0)` Å. Same H₂ preset as `tools/pyscf_oracle/pin_scf.py`.

## Electrostatics (binding convention)

**Truncated lattice sum, no Ewald.** Image half-count per axis is
`m_d = floor(rcut / L_d)` with `rcut = 2 √(−ln(10⁻⁸) / α_min)` from the
most diffuse primitive. J, K, S, T, V and `E_nn` share that window.

On this 20 Å cell, STO-3G H has `rcut ≈ 11 Å < 20 Å`, so **m = 0**:
only T = 0. The energy is the molecular RHF energy.

## Oracle

- **Code:** PySCF 2.14.0 `pbc.gto` + `pbc.scf.RHF` (offline; never in ctest)
- **Date:** 2026-08-17
- **Command:**

```bash
python3 - <<'PY'
from pyscf import gto, scf as molscf
from pyscf.pbc import gto as pgto, scf as pscf
import numpy as np
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
            unit='Angstrom', cart=True, verbose=0)
print('mol', molscf.RHF(mol).kernel())
cell = pgto.M(atom='H 0 0 0; H 0 0 0.74', a=20.0*np.eye(3),
              basis='sto-3g', unit='Angstrom', cart=True, verbose=0)
print('pbc', pscf.RHF(cell).kernel())
PY
```

- **Printed molecular RHF:** `-1.116759307396` Ha (in-tree twin:
  `regressions/scf/gf2-h2-sto3g`)
- **PBC 20 Å:** the truncated ATV window has no extra images, so the
  pin is this molecular number. Re-run the command to confirm PySCF
  `pbc.scf.RHF` on the same cell agrees to ≤ 1e-5 Ha (Ewald on a
  neutral molecule in a large vacuum box).
- **Tolerance:** \(1\times 10^{-5}\) Ha

## Product path

`cpu::scf::RHF{BasisSet::STO3G}` + `rhf.periodic = true` + `run(frame)`.
`periodic` defaults to false and does not touch the molecular Fock path.
