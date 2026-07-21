# Ar / def2-TZVP RHF x2c-1e

## System

- Atom: Ar (Z = 18) at the origin
- Basis: def2-TZVP (spherical, all-electron; Z ≤ 18 covered)
- Method: RHF, closed shell (18 electrons)

## Oracle

PySCF 2.14.0, contracted AO basis (`xuncontract=False` — **required**;
PySCF's default `xuncontract=True` path is out of scope):

```python
from pyscf import gto, scf
mol = gto.M(atom='Ar 0 0 0', basis='def2-TZVP', verbose=0)
e_nr = scf.RHF(mol).run(conv_tol=1e-12).e_tot
mf = scf.RHF(mol).x2c1e()
mf.with_x2c.xuncontract = False
e_x2c = mf.run(conv_tol=1e-12).e_tot
```

| Quantity | Value [Ha] |
|---|---|
| E_nonrel | −526.8027603325265 |
| E_x2c    | −528.6442231685685 |
| Δ = E_x2c − E_nonrel | −1.8414628360419556 |

## Acceptance

- `|E_x2c − pin| ≤ 1e-5 Ha`
- `E_x2c − E_nonrel < 0` and `|Δ|` within 5% of the pinned PySCF difference

## Literature

- Kutzelnigg, Liu, *J. Chem. Phys.* **123**, 241102 (2005), DOI 10.1063/1.2137315
- Peng, Reiher, *Theor. Chem. Acc.* **131**, 1081 (2012), DOI 10.1007/s00214-011-1081-y
