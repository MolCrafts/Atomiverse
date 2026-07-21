# H2O / def2-TZVP RI-JK — external oracle pins

Case: RHF with RI-JK density fitting over the def2 universal JKFIT
auxiliary basis, double-gated against (1) our own exact-ERI RHF and
(2) the externally pinned PySCF `density_fit()` energy.

## Literature basis

- Robust Coulomb-metric fit (error second order in the fitting
  residual): O. Vahtras, J. Almlöf, M. W. Feyereisen,
  *Chem. Phys. Lett.* **213**, 514 (1993),
  DOI [10.1016/0009-2614(93)89151-7](https://doi.org/10.1016/0009-2614(93)89151-7).
- RI-K and the universal JKFIT auxiliary basis: F. Weigend,
  *Phys. Chem. Chem. Phys.* **4**, 4285 (2002),
  DOI [10.1039/B204199P](https://doi.org/10.1039/B204199P).
- Accuracy of the Coulomb fit / auxiliary basis matching: F. Weigend,
  *Phys. Chem. Chem. Phys.* **8**, 1057 (2006),
  DOI [10.1039/b515623h](https://doi.org/10.1039/b515623h).

## Oracle runs (PySCF 2.14.0, `~/work/.x86_64/bin/python`, 2026-07-21)

Command, verbatim:

```python
import numpy as np
from pyscf import gto, scf, df

mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
            basis='def2-tzvp')  # spherical (PySCF default, cart=False)
mf = scf.RHF(mol); mf.conv_tol = 1e-12
e_exact = mf.kernel()
mfd = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
mfd.conv_tol = 1e-12
e_df = mfd.kernel()
print('H2O/def2-TZVP  E_exact  = %.12f' % e_exact)
print('H2O/def2-TZVP  E_DF-JK  = %.12f' % e_df)
print('DF - exact = %.3e Ha' % (e_df - e_exact))
```

Output:

```
H2O/def2-TZVP  E_exact  = -76.059004760577
H2O/def2-TZVP  E_DF-JK  = -76.058999347289
DF - exact = 5.413e-06 Ha
```

## Pinned values and gates

| Quantity | Value | Gate |
|---|---|---|
| PySCF exact-ERI RHF | −76.059004760577 Ha | (context only) |
| PySCF DF (`density_fit`, def2-universal-jkfit) | **−76.058999347289 Ha** | our E(RI-JK) within **1e-6 Ha** |
| PySCF fit error (DF − exact) | +5.413e-6 Ha | our \|E(RI-JK) − E(own exact)\| ≤ **5e-5 Ha** |

Gate 1 (≤ 50 μHa vs our own exact-ERI HF) directly tests the
second-order-error claim of the Coulomb-metric robust fit with a
matched auxiliary basis; PySCF's own fit error (5.4 μHa) shows the
expected order of magnitude. Gate 2 (≤ 1e-6 Ha vs the PySCF DF pin)
proves the fit lives in the same *spherical* def2-universal-jkfit
auxiliary span — a Cartesian-span fit would pass gate 1 but drift off
gate 2.

Geometry (Å): O(0, 0, 0.1173), H(0, ±0.7572, −0.4692) — identical to
regressions/scf/h2o_rhf_sto3g.
