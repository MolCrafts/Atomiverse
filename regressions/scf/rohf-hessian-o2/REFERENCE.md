# Reference — O₂ ROHF/STO-3G nuclear Hessian

## Method

Nuclear Hessian of the ROHF energy via central finite difference of the
**analytic** ROHF nuclear forces (`ROHF::forces` / ROHF Lagrangian),
HF/STO-3G on triplet O₂ (charge 0, \(2S=2\)). Public API:
`cpu::scf::ROHF{9,7}.hessian(frame)`. Observable is the dense Cartesian
Hessian in eV/Å² (Frame/vib contract). Full three-space ROHF CPHF is a
follow-up; PySCF 2.14.0 also has no `hessian.ROHF`.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
|---|---|---|---|
| O | 0.0 | 0.0 | 0.0 |
| O | 0.0 | 0.0 | 1.2075 |

20 Å cubic box (gas phase; the box does not enter the molecular SCF).
Same geometry as `o2_rohf_sto3g`, `o2_rohf_grad`, `uhf-hessian-o2`.

## Pinned oracle (offline)

PySCF **2.14.0** (`~/work/.x86_64/bin/python`) does **not** implement
`hessian.ROHF` / `mf.Hessian()` for ROHF (`NotImplementedError`). The pin is
the central finite-difference of the **analytic** ROHF nuclear gradient
(`mf.nuc_grad_method().kernel()`), step \(h = 10^{-4}\) Bohr, symmetrized,
converted to eV/Å² with Atomiverse CODATA factors
(`kHartreeInEv = 27.211386245988`, `kAngstromInBohr = 1.8897261245650618`).
Run 2026-07-23:

```python
import numpy as np
from pyscf import gto, scf

HA_TO_EV = 27.211386245988
ANG_IN_BOHR = 1.8897261245650618
BOHR_IN_A = 1.0 / ANG_IN_BOHR
to_ev_a2 = HA_TO_EV / (BOHR_IN_A**2)

mol0 = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=2,
             cart=True, unit="Angstrom", verbose=0)
mf0 = scf.ROHF(mol0)
mf0.conv_tol = 1e-12
e0 = mf0.kernel()                       # -147.632166990682 Ha
coords = mol0.atom_coords().copy()      # Bohr
h = 1e-4
H = np.zeros((2, 2, 3, 3))
for A in range(2):
    for i in range(3):
        def grad_at(delta):
            c = coords.copy(); c[A, i] += delta
            mol = mol0.copy(); mol.set_geom_(c, unit="Bohr"); mol.build(0, 0)
            mf = scf.ROHF(mol); mf.conv_tol = 1e-12; mf.kernel()
            return mf.nuc_grad_method().kernel()
        gp, gm = grad_at(+h), grad_at(-h)
        H[A, :, i, :] = (gp - gm) / (2 * h)
flat = np.zeros((6, 6))
for A in range(2):
    for i in range(3):
        for B in range(2):
            for j in range(3):
                flat[3*A+i, 3*B+j] = H[A, B, i, j]
flat = 0.5 * (flat + flat.T)
H_ev = flat * to_ev_a2
```

Offline helper:

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python - <<'PY'
# (script above)
PY
```

Pinned values at this geometry (eV/Å²). Off-axis elements are numerically
zero; only the Cartesian-diagonal / stretch blocks are product-gated.

| Element (row, col) | atoms/axes | Value [eV/Å²] |
|---|---|---|
| \(H_{xx}(0,0)\) | O1-x / O1-x | −1.1102153966 |
| \(H_{yy}(1,1)\) | O1-y / O1-y | −1.1102153968 |
| \(H_{zz}(2,2)\) | O1-z / O1-z | +135.5737365643 |
| \(H_{xx}(0,3)\) | O1-x / O2-x | +1.1102153967 |
| \(H_{yy}(1,4)\) | O1-y / O2-y | +1.1102153968 |
| \(H_{zz}(2,5)\) | O1-z / O2-z | −135.5737365648 |
| \(H_{xx}(3,3)\) | O2-x / O2-x | −1.1102153969 |
| \(H_{yy}(4,4)\) | O2-y / O2-y | −1.1102153968 |
| \(H_{zz}(5,5)\) | O2-z / O2-z | +135.5737365643 |

Reference energy (not the product gate): \(E_{\mathrm{ROHF}} = -147.632166990682\) Ha.

## Tolerance and its rationale

**1e-5 eV/Å² per gated element.** Atomiverse uses the same algorithm class as
the oracle (FD of analytic ROHF gradients/forces, \(h = 10^{-4}\) Bohr).
Stretch blocks are \(\sim 10^2\) eV/Å²; soft transverse blocks \(\sim 1\) eV/Å².
1e-5 catches step/sign/unit errors. Off-axis elements asserted
\(\lvert H_{ij}\rvert < 10^{-3}\).

## Performance

**`perf_exempt=true`.** PySCF 2.14.0 has no fair analytic ROHF Hessian twin
(`mf.Hessian()` raises `NotImplementedError`; no `hessian.ROHF`). The offline
oracle is FD of analytic gradients (many ROHF SCFs), which is the same work
class as Atomiverse `ROHF::hessian` and is not a one-shot analytic CPHF twin
for a fair wall-time ratio.

```bash
# Confirm no PySCF analytic ROHF Hessian:
~/work/.x86_64/bin/python -c "
from pyscf import gto, scf
m=gto.M(atom='O 0 0 0; O 0 0 1.2075',basis='sto-3g',spin=2,cart=True)
mf=scf.ROHF(m); mf.kernel(); mf.Hessian()
"
# → NotImplementedError: Method Hessian invalid or not implemented
```

`ratio_atv_over_pyscf`: n/a (perf_exempt). `within_10x`: n/a (exempt).

## References

- C. C. J. Roothaan, *Rev. Mod. Phys.* **32**, 179 (1960) — ROHF.
- Guest & Saunders, *Mol. Phys.* **28**, 819 (1974) — coupling convention.
- J. A. Pople et al., *Int. J. Quantum Chem. Symp.* **13**, 225 (1979) —
  analytic HF second derivatives.
- Y. Yamaguchi et al., *A New Dimension to Quantum Chemistry* (Oxford, 1994).

## What is checked

`cpu::scf::ROHF{9,7}` → `rohf.hessian(frame)` returns a 6×6 Hessian in eV/Å².
Product gate: the nine Cartesian-diagonal / stretch elements above within
**1e-5 eV/Å²** of the offline pins; failure prints got / want / tol and exits
non-zero. ≤ 60 s, single node.
