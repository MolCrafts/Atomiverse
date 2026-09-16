# Reference — O₂ UKS-LSDA/STO-3G nuclear Hessian

## Method

Nuclear Hessian of the unrestricted Kohn–Sham energy via central finite
difference of the **analytic** UKS nuclear forces (`UKS::forces` / KS
gradient with grid-response XC), LSDA (Slater–VWN5) / STO-3G on triplet O₂
(\(2S=2\), \(n_\alpha=9\), \(n_\beta=7\)). Public API:
`cpu::scf::UKS{9,7,XCType::LSDA}.hessian(frame, &guess)` with ROKS density
warm-start (same multi-minima path as `uks-tda-o2-lsda`). Observable is the
dense Cartesian Hessian in eV/Å² (Frame/vib contract).

Full analytic UCPKS (PySCF `hessian.UKS` closed form with nuclear XC second
derivatives) is a follow-up; the offline product pin uses the **same
algorithm class** as Atomiverse (FD of analytic UKS gradients), matching
the RKS / ROHF Hessian product path. PySCF's analytic
`hessian.uks.Hessian(mf).kernel()` differs from this FD twin by
\(\lesssim 3.5\cdot 10^{-2}\) eV/Å² at \(h=10^{-4}\) Bohr (recorded below
for cross-check only).

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
|---|---|---|---|
| O | 0.0 | 0.0 | 0.0 |
| O | 0.0 | 0.0 | 1.2075 |

20 Å cubic box (gas phase; the box does not enter the molecular SCF).
Same geometry as `uks-tda-o2-lsda`, `uhf-hessian-o2`, `ump2-o2-sto3g`.

## Pinned PySCF oracle (FD of analytic UKS gradients)

PySCF **2.14.0** (`~/work/.x86_64/bin/python` →
`/nobackup/proj/disk/teoroo/personal/jicli594/work/.x86_64/bin/python`),
run 2026-07-24. Central FD of `mf.nuc_grad_method().kernel()`, step
\(h = 10^{-4}\) Bohr, symmetrized, converted to eV/Å² with Atomiverse
CODATA factors (`kHartreeInEv = 27.211386245988`,
`kBohrInAngstrom = 0.529177210903`):

```python
import numpy as np
from pyscf import gto, dft

HA_TO_EV = 27.211386245988
BOHR_IN_A = 0.529177210903
to_ev_a2 = HA_TO_EV / (BOHR_IN_A**2)

mol0 = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=2,
             cart=True, unit="Angstrom", verbose=0)
mf0 = dft.UKS(mol0); mf0.xc = "LDA,VWN"; mf0.conv_tol = 1e-12
e0 = mf0.kernel()                       # -147.193805922283 Ha
coords = mol0.atom_coords().copy()
h = 1e-4
H = np.zeros((2, 2, 3, 3))
for A in range(2):
    for i in range(3):
        def g_at(delta, A=A, i=i):
            c = coords.copy(); c[A, i] += delta
            mol = mol0.copy(); mol.set_geom_(c, unit="Bohr"); mol.build(0, 0)
            mf = dft.UKS(mol); mf.xc = "LDA,VWN"; mf.conv_tol = 1e-12
            mf.kernel()
            return mf.nuc_grad_method().kernel()
        gp, gm = g_at(+h), g_at(-h)
        H[A, :, i, :] = (gp - gm) / (2 * h)
flat = np.zeros((6, 6))
for A in range(2):
    for i in range(3):
        for B in range(2):
            for j in range(3):
                flat[3 * A + i, 3 * B + j] = H[A, B, i, j]
flat = 0.5 * (flat + flat.T)
H_ev = flat * to_ev_a2
```

Cross-check (not the product gate): PySCF analytic
`from pyscf.hessian import uks as hess_uks; hess_uks.Hessian(mf0).kernel()`
at the same geometry differs from this FD twin by at most
\(3.44\cdot 10^{-2}\) eV/Å² (max element abs). Analytic stretch
\(H_{zz}(0,0) \approx 128.490\) eV/Å² vs FD \(128.456\) eV/Å².

Reference energy (not the product gate): \(E_{\mathrm{UKS}} = -147.193805922283\) Ha.

Gated Cartesian elements (eV/Å²) — FD twin pin:

| (row, col) | atoms/axes | Value [eV/Å²] |
|---|---|---|
| \(H_{xx}(0,0)\) | O1-x / O1-x | −6.951874310841 |
| \(H_{yy}(1,1)\) | O1-y / O1-y | −6.951874310747 |
| \(H_{zz}(2,2)\) | O1-z / O1-z | +128.455930123774 |
| \(H_{xx}(0,3)\) | O1-x / O2-x | +6.951874311534 |
| \(H_{yy}(1,4)\) | O1-y / O2-y | +6.951874311029 |
| \(H_{zz}(2,5)\) | O1-z / O2-z | −128.455930112985 |
| \(H_{xx}(3,3)\) | O2-x / O2-x | −6.951874311765 |
| \(H_{yy}(4,4)\) | O2-y / O2-y | −6.951874311352 |
| \(H_{zz}(5,5)\) | O2-z / O2-z | +128.455930103923 |

Off-axis elements are numerically zero on this collinear geometry
(\(\lesssim 10^{-8}\) eV/Å² in the PySCF FD twin).

## Tolerance and its rationale

**2e-2 eV/Å² per gated element.** Atomiverse and the oracle share the same
algorithm class (FD of analytic UKS gradients/forces, \(h = 10^{-4}\) Bohr),
but the KS integration grids are not bit-identical (Atomiverse
`MatchPySCF` vs PySCF `level=3`). That residual is visible on the analytic
forces and propagates through the FD step into \(\sim 10^{-2}\) eV/Å² on
Hessian elements — the same floor as `rks-hessian-h2o-sto3g`. The product
gate is set at \(2\cdot 10^{-2}\) (2× that floor) so it still catches
sign/factor errors on stiff stretch blocks (\(\sim 10^2\) eV/Å²) without
demanding bit-identical grids. A full UCPKS twin of PySCF `hessian.UKS` at
1e-5 eV/Å² is a follow-up.

Off-axis elements are asserted only to be \(\lvert H_{ij}\rvert < 10^{-2}\)
(near zero by symmetry).

## Fair wall-time (offline; not ctest)

Fair twin: Atomiverse `UKS::hessian` (12 force evaluations after ROKS
warm-start + reference) vs PySCF loop of 12 UKS analytic gradients (same FD
skeleton, \(h=10^{-4}\) Bohr). Analytic `hessian.UKS` is a different
algorithm (closed form) and is **not** the fair wall twin for this product
path.

Recorded 2026-07-24 (`OMP_NUM_THREADS=8`, `OMP_PROC_BIND=close`,
Release build; median of 3 runs):

| Side | Median wall [s] |
|---|---|
| Atomiverse `reg_uks-hessian-o2-sto3g` | 1.904 |
| PySCF FD twin (ref + 12× UKS + `nuc_grad` + restore) | 0.894 |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf,FD}}\) | **2.13** |
| within_10x | **true** |
| PySCF analytic `hessian.UKS` (cross-check only) | 0.177 |
| ratio vs analytic kernel (unfair) | 10.8 |

## References

- B. G. Johnson, J. A. Pople, P. M. W. Gill, M. J. Frisch,
  *Chem. Phys. Lett.* **220**, 377 (1994) — analytic DFT second derivatives
  (target closed form; this product ships FD of analytic forces).
- Y. Yamaguchi et al., *A New Dimension to Quantum Chemistry* (Oxford, 1994)
  — unrestricted Hessian framework.
- PySCF `hessian.uks` / `grad.uks` — external oracle engines.

## What is checked

`cpu::scf::UKS{9,7,XCType::LSDA}` with ROKS warm-start, `basis = STO3G`,
`tol = 1e-12`, `grid_level = MatchPySCF` → `uks.hessian(frame, &guess)`
returns a 6×6 Hessian in eV/Å². Product gate: the nine Cartesian-diagonal /
stretch elements above within **2e-2 eV/Å²** of the pinned PySCF FD twin;
failure prints got / want / tol and exits non-zero. ≤ 60 s, single node.
