# scf/periodic-hf

One case; every section below is run by `atv_regressions --case scf/periodic-hf` and keeps its own oracle record.


## `periodic-rhf-gamma`

H₂ / STO-3G Cartesian **Γ-point periodic RHF** in a 20 Å cubic cell.

### Geometry

| Atom | x / Å | y / Å | z / Å |
| ---- | ----- | ----- | ----- |
| H    | 0     | 0     | 0     |
| H    | 0     | 0     | 0.74  |

`Frame.box = cubic(20.0)` Å. Same H₂ preset as `tools/pyscf_oracle/pin_scf.py`.

### Electrostatics (binding convention)

**Truncated lattice sum, no Ewald.** Image half-count per axis is
`m_d = floor(rcut / L_d)` with `rcut = 2 √(−ln(10⁻⁸) / α_min)` from the
most diffuse primitive. J, K, S, T, V and `E_nn` share that window.

On this 20 Å cell, STO-3G H has `rcut ≈ 11 Å < 20 Å`, so **m = 0**:
only T = 0. The energy is the molecular RHF energy.

### Oracle

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
  `regressions/scf/green-function`)
- **PBC 20 Å:** the truncated ATV window has no extra images, so the
  pin is this molecular number. Re-run the command to confirm PySCF
  `pbc.scf.RHF` on the same cell agrees to ≤ 1e-5 Ha (Ewald on a
  neutral molecule in a large vacuum box).
- **Tolerance:** \(1\times 10^{-5}\) Ha

### Product path

`cpu::scf::RHF{BasisSet::STO3G}` + `rhf.periodic = true` + `run(frame)`.
`periodic` defaults to false and does not touch the molecular Fock path.


## periodic-kpoint-hf

H₂ / STO-3G in a cubic 4 Bohr cell. The cell edge is smaller than the
Gaussian pair reach (`cell.rcut` ≈ 16.51 Bohr at `precision = 1e-8`), so
lattice images of the basis overlap the home cell.

Monkhorst–Pack meshes use PySCF `make_kpts` (`wrap_around=False`).
H–H lies on c, so `k_mesh = {2,1,1}` samples the bond-perpendicular
direction and `{1,1,2}` samples along the bond.

### Oracle

PySCF 2.14.0 `pyscf.pbc.scf`, default `FFTDF`, `exxdiv='ewald'`.

```
python tools/pyscf_oracle/pin_scf.py krhf
python tools/pyscf_oracle/pin_scf.py krhf --kmesh 2,1,1
python tools/pyscf_oracle/pin_scf.py kuhf --kmesh 2,1,1 --spin 2
```

| Case | Call | E [Ha] |
| --- | --- | --- |
| KRHF `{1,1,2}` | `KRHF` | −1.3541438798330374 |
| KRHF `{2,1,1}` | `KRHF` | −1.4471748215213558 |
| triplet KUHF `{2,1,1}` | `KUHF`, `cell.spin = 2` | −0.7658470471886634 |

`KUHF.nelec` (`pyscf/pbc/scf/kuhf.py`) is the mesh total
`(tot_electrons(nk) + spin)//2`. `tot_electrons(2) = 4` and `spin = 2`
give Nα = 3, Nβ = 1. Spin is not multiplied by the number of k-points.
The same call prints `enuc = -0.6281049380287604` Ha
(`pyscf.pbc.gto.cell.ewald`).

### Gate

`|E - E_pin| ≤ 1e-6` Ha. The lattice-sum cutoff on S and T
(PySCF `cell.rcut`) is the leading discrepancy versus libcint
`int1e_*`; J and K on a matched AO grid agree much more tightly.

### Not this case

Analytic nuclear forces (`krhf_nuclear_forces`, `kuhf_nuclear_forces`).
Shells with l > 0 (`krhf_ao_lgt0`). Device KUHF.
