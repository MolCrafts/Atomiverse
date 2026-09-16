# Reference — H₂O RKS-LSDA/STO-3G nuclear Hessian

## Method

Nuclear Hessian of the restricted Kohn–Sham energy via central finite
difference of the **analytic** RKS nuclear forces (`RKS::forces` / KS
gradient with grid-response XC), LSDA (Slater–VWN5) / STO-3G on water.
Public API: `cpu::scf::RKS{XCType::LSDA}.hessian(frame)`. Observable is the
dense Cartesian Hessian in eV/Å² (Frame/vib contract).

Full analytic CPKS (PySCF `hessian.RKS` closed form with nuclear XC second
derivatives) is a follow-up; the offline product pin uses the **same
algorithm class** as Atomiverse (FD of analytic RKS gradients), matching
the ROHF Hessian product path. PySCF's analytic `mf.Hessian().kernel()`
differs from this FD twin by \(\lesssim 4\cdot 10^{-2}\) eV/Å² at
\(h=10^{-4}\) Bohr (recorded below for cross-check only).

## Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

20 Å cubic box (gas phase; the box does not enter the molecular SCF).

## Pinned PySCF oracle (FD of analytic RKS gradients)

PySCF **2.14.0** (`~/work/.x86_64/bin/python`), run 2026-07-23. Central FD
of `mf.nuc_grad_method().kernel()`, step \(h = 10^{-4}\) Bohr, symmetrized,
converted to eV/Å² with Atomiverse CODATA factors
(`kHartreeInEv = 27.211386245988`, `kBohrInAngstrom = 0.529177210903`):

```python
import numpy as np
from pyscf import gto, dft

HA_TO_EV = 27.211386245988
BOHR_IN_A = 0.529177210903
to_ev_a2 = HA_TO_EV / (BOHR_IN_A**2)

mol0 = gto.M(atom="""
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""", basis="sto-3g", cart=True, unit="Angstrom", verbose=0)
mf0 = dft.RKS(mol0); mf0.xc = "LDA,VWN"; mf0.conv_tol = 1e-12
e0 = mf0.kernel()                       # -74.732038853874 Ha
coords = mol0.atom_coords().copy()
h = 1e-4
H = np.zeros((3, 3, 3, 3))
for A in range(3):
    for i in range(3):
        def g_at(delta, A=A, i=i):
            c = coords.copy(); c[A, i] += delta
            mol = mol0.copy(); mol.set_geom_(c, unit="Bohr"); mol.build(0, 0)
            mf = dft.RKS(mol); mf.xc = "LDA,VWN"; mf.conv_tol = 1e-12
            mf.kernel()
            return mf.nuc_grad_method().kernel()
        gp, gm = g_at(+h), g_at(-h)
        H[A, :, i, :] = (gp - gm) / (2 * h)
flat = np.zeros((9, 9))
for A in range(3):
    for i in range(3):
        for B in range(3):
            for j in range(3):
                flat[3 * A + i, 3 * B + j] = H[A, B, i, j]
flat = 0.5 * (flat + flat.T)
H_ev = flat * to_ev_a2
```

Cross-check (not the product gate): PySCF analytic `mf.Hessian().kernel()`
at the same geometry differs from this FD twin by at most
\(4.34\cdot 10^{-2}\) eV/Å² (max element abs).

Reference energy (not the product gate): \(E_{\mathrm{RKS}} = -74.732038853874\) Ha.

Gated Cartesian elements (eV/Å²) — FD twin pin:

| (row, col) | Value [eV/Å²] |
|---|---|
| \(H_{xx}(0,0)\) | −9.632120712096 |
| \(H_{yy}(1,1)\) | +89.527387528595 |
| \(H_{zz}(2,2)\) | +54.829425918251 |
| \(H_{xx}(0,3)\) | +4.820104284662 |
| \(H_{xx}(0,6)\) | +4.820104284678 |
| \(H_{yy}(1,4)\) | −44.758907247881 |
| \(H_{yz}(1,5)\) | +38.397479554108 |
| \(H_{yy}(1,7)\) | −44.758906500277 |
| \(H_{yz}(1,8)\) | −38.397479301244 |
| \(H_{zz}(2,5)\) | −27.417403943596 |
| \(H_{yy}(4,4)\) | +47.406667576594 |
| \(H_{zz}(5,5)\) | +26.895192808433 |
| \(H_{yy}(7,7)\) | +47.406666805220 |
| \(H_{zz}(8,8)\) | +26.895192389626 |

## Tolerance and its rationale

**2e-2 eV/Å² per gated element.** Atomiverse and the oracle share the same
algorithm class (FD of analytic RKS gradients/forces, \(h = 10^{-4}\) Bohr),
but the KS integration grids are not bit-identical: Atomiverse
`MatchPySCF` builds 33632 points on this water/STO-3G case while PySCF
`level=3` builds 33704. That residual is visible as \(\sim 2\cdot 10^{-3}\)
eV/Å on the analytic forces and propagates through the FD step into
\(\sim 10^{-2}\) eV/Å² on Hessian elements. The product gate is set at
\(2\cdot 10^{-2}\) (2× that floor) so it still catches sign/factor errors
on stiff blocks (\(\sim 10^2\) eV/Å²) without demanding bit-identical grids.
A full CPKS twin of PySCF `hessian.RKS` at 1e-5 eV/Å² is a follow-up.

## Fair wall-time (offline; not ctest)

Fair twin: Atomiverse `RKS::hessian` (18 force evaluations) vs PySCF loop of
18 RKS analytic gradients (same FD skeleton, \(h=10^{-4}\) Bohr). Analytic
`hessian.RKS` is a different algorithm (closed form, median ~0.094 s) and is
**not** the fair wall twin for this product path.

Recorded 2026-07-23 (`OMP_NUM_THREADS=8`, Release build):

| Side | Median wall [s] |
|---|---|
| Atomiverse `reg_rks-hessian-h2o-sto3g` | 3.138 |
| PySCF FD twin (18× RKS + `nuc_grad`) | 1.019 |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf,FD}}\) | **3.08** |
| within_10x | **true** |
| PySCF analytic `Hessian().kernel()` (cross-check only) | 0.094 |
| ratio vs analytic kernel (unfair) | 33.4 |

## References

- B. G. Johnson, J. A. Pople, P. M. W. Gill, M. J. Frisch,
  *Chem. Phys. Lett.* **220**, 377 (1994) — analytic DFT second derivatives
  (target closed form; this product ships FD of analytic forces).
- PySCF `hessian.rks` / `grad.rks` — external oracle engines.

## What is checked

`cpu::scf::RKS{XCType::LSDA}` with `basis = STO3G`, `tol = 1e-12` →
`rks.hessian(frame)` returns a 9×9 Hessian in eV/Å². Product gate: gated
elements above within **2e-2 eV/Å²** of the pinned PySCF FD twin; failure
prints got / want / tol and exits non-zero. ≤ 60 s, single node.
