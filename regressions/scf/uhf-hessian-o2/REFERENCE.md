# Reference — O₂ UHF/STO-3G analytic nuclear Hessian

## Method

Analytic UHF nuclear Hessian via unrestricted coupled-perturbed Hartree–Fock
(CPHF) linear response, HF/STO-3G on triplet O₂ (charge 0, \(2S=2\)). Public
API: `cpu::scf::UHF{9,7}.hessian(frame, &guess)` with ROHF density warm-start
(same multi-minima path as `ump2-o2-sto3g` / `o2_rohf_sto3g`). Observable is
the dense Cartesian Hessian in eV/Å² (Frame/vib contract).

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
|---|---|---|---|
| O | 0.0 | 0.0 | 0.0 |
| O | 0.0 | 0.0 | 1.2075 |

20 Å cubic box (gas phase; the box does not enter the molecular SCF).
Same geometry as `o2_rohf_sto3g`, `ump2-o2-sto3g`, `o2_rohf_grad`.

## Pinned PySCF oracle

PySCF **2.14.0** (`~/work/.x86_64/bin/python`), run 2026-07-23, commands
verbatim:

```python
import numpy as np
from pyscf import gto, scf, hessian

mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=2,
            cart=True, unit="Angstrom", verbose=0)
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
e = mf.kernel()                       # -147.633946785450 Ha
h = hessian.UHF(mf).kernel()          # (natm,natm,3,3) Ha/Bohr^2
# Flatten to [3N,3N] and convert to eV/Å²:
#   HA_TO_EV = 27.211386245988
#   BOHR_IN_A = 0.529177210903
#   H_ev = H_ha * HA_TO_EV / BOHR_IN_A**2
```

Offline pin helper (optional extension):

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python - <<'PY'
# (script above)
PY
```

Pinned values at this geometry (eV/Å²). Off-axis elements are numerically
zero (\(\lesssim 10^{-14}\)); only the Cartesian-diagonal / stretch blocks
are product-gated.

| Element (row, col) | atoms/axes | Value [eV/Å²] |
|---|---|---|
| \(H_{xx}(0,0)\) | O1-x / O1-x | −1.0567201218 |
| \(H_{yy}(1,1)\) | O1-y / O1-y | −1.0567201218 |
| \(H_{zz}(2,2)\) | O1-z / O1-z | +135.7176127486 |
| \(H_{xx}(0,3)\) | O1-x / O2-x | +1.0567201218 |
| \(H_{yy}(1,4)\) | O1-y / O2-y | +1.0567201218 |
| \(H_{zz}(2,5)\) | O1-z / O2-z | −135.7176127486 |
| \(H_{xx}(3,3)\) | O2-x / O2-x | −1.0567201219 |
| \(H_{yy}(4,4)\) | O2-y / O2-y | −1.0567201220 |
| \(H_{zz}(5,5)\) | O2-z / O2-z | +135.7176127485 |

Reference energy (not the product gate): \(E_{\mathrm{UHF}} = -147.633946785450\) Ha.

One real stretch frequency from the same Hessian (thermo analysis, not
gated): \(\nu \approx 2147.91\) cm⁻¹.

## Tolerance and its rationale

**1e-5 eV/Å² per gated element.** Analytic UHF CPHF and PySCF `hessian.UHF`
evaluate the same closed-form second derivatives; residual differences are
SCF/CPHF convergence noise. The stretch blocks are \(\sim 10^2\) eV/Å², so
1e-5 is a relative \(\sim 10^{-7}\) band — tight enough to catch a sign or
spin-channel error in the response or skeleton 2e pieces. Off-axis elements
are asserted only to be \(\lvert H_{ij}\rvert < 10^{-3}\) (near zero by
symmetry on this collinear geometry).

## References

- J. A. Pople, R. Krishnan, H. B. Schlegel, J. S. Binkley,
  *Int. J. Quantum Chem. Symp.* **13**, 225 (1979),
  DOI 10.1002/qua.560160825 — CPHF and analytic HF second derivatives.
- Y. Yamaguchi, Y. Osamura, J. D. Goddard, H. F. Schaefer,
  *A New Dimension to Quantum Chemistry: Analytic Derivative Methods in
  Ab Initio Molecular Electronic Structure Theory* (Oxford, 1994) —
  unrestricted Hessian.
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), response-theory chapter.

## What is checked

`cpu::scf::UHF{9,7}` with ROHF warm-start → `uhf.hessian(frame, &guess)`
returns a 6×6 Hessian in eV/Å². Product gate: the nine Cartesian-diagonal /
stretch elements above within **1e-5 eV/Å²** of the pinned PySCF values;
failure prints got / want / tol and exits non-zero. ≤ 60 s, single node.
