# Reference — H₂O HF/6-31G* harmonic frequencies (analytic CPHF Hessian)

## Method

Analytic RHF nuclear Hessian via coupled-perturbed Hartree-Fock (CPHF)
linear response, HF/6-31G* (`Pople631Gs`, Cartesian d functions — the
Pople convention, PySCF `cart=True`), fed through the existing vib
pipeline (`cpu::vib::MassWeighting` → `cpu::vib::HessianDiagonalizer`) to
harmonic frequencies. Two gates:

1. **First gate (in-house self-consistency):** the analytic frequencies
   against the finite-difference production line — the `cpu::vib::Hessian`
   driver (central differences of the analytic forces, step 10⁻³ Å) with
   the same `cpu::scf::RHF` bound at `vib::Phase::force` — mode-by-mode
   ≤ 1 cm⁻¹. Both producers are in-house; they meet at the same
   `vib::HessianResult` consumer.
2. **External oracle:** the pinned PySCF `hessian.RHF` frequencies below,
   mode-by-mode ≤ 1 cm⁻¹.

## Geometry

The repository's standard water case geometry (identical to
`regressions/scf/h2o_rhf_sto3g`, `h2o_rhf_dipole`, `h2o_charges_pyscf`):
R(O–H) = 0.95 Å, ∠H–O–H = 104.5°; O at the origin, both H symmetric about
+z in the xz-plane (half-angle 52.25°):

| Atom | x [Å] | y [Å] | z [Å] |
|---|---|---|---|
| O | 0.00000000 | 0.0 | 0.00000000 |
| H | +0.75115510 | 0.0 | 0.58160642 |
| H | −0.75115510 | 0.0 | 0.58160642 |

20 Å cubic box (gas phase; the box does not enter the molecular SCF).
Note this is NOT the equilibrium HF/6-31G* geometry, so the frequencies
below are *this geometry's* harmonic frequencies, not literature
equilibrium values — the oracle is the pinned external run at exactly
this geometry.

## Pinned PySCF oracle

PySCF **2.14.0** (`~/work/.x86_64/bin/python`), run 2026-07-21, commands
verbatim:

```python
import numpy as np, math
from pyscf import gto, scf, hessian
from pyscf.hessian import thermo

r_oh = 0.95
half = 0.5 * 104.5 * math.pi / 180.0
hx = r_oh * math.sin(half); hz = r_oh * math.cos(half)
mol = gto.M(atom=f"O 0 0 0; H {hx:.16f} 0 {hz:.16f}; H {-hx:.16f} 0 {hz:.16f}",
            basis="6-31g*", cart=True, unit="Angstrom", verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
h = hessian.RHF(mf).kernel()
res = thermo.harmonic_analysis(mol, h)
print(mf.e_tot)                  # -76.0107068004 Ha
print(res['freq_wavenumber'])    # [1843.7084 4036.1737 4146.3782] cm^-1
```

Pinned values at this geometry:

| Quantity | Value |
|---|---|
| E_RHF/6-31G* (cart) | −76.0107068004 Ha |
| ν₁ (bend, a₁) | 1843.7084 cm⁻¹ |
| ν₂ (sym stretch, a₁) | 4036.1737 cm⁻¹ |
| ν₃ (asym stretch, b₂) | 4146.3782 cm⁻¹ |

## Tolerance and its rationale

**1 cm⁻¹ per mode, both gates.** The analytic Hessian and PySCF's
`hessian.RHF` evaluate the same closed-form CPHF expressions; residual
differences are SCF/CPHF convergence noise and the harmonic-analysis
projection, well below 1 cm⁻¹. The FD gate carries the additional
O(h²) truncation of the central difference at 10⁻³ Å, also ≪ 1 cm⁻¹ for
water. This is a deliberately tight parity band — a sign/factor error in
any CPHF piece (skeleton derivatives, B^x, the response coupling) moves
frequencies by tens to hundreds of cm⁻¹.

## References

- J. A. Pople, R. Krishnan, H. B. Schlegel, J. S. Binkley,
  *Int. J. Quantum Chem. Symp.* **13**, 225 (1979),
  DOI 10.1002/qua.560160825 — CPHF and analytic HF second derivatives.
- P. Pulay, *Mol. Phys.* **17**, 197 (1969) — forces and derivative
  methods.
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure
  Theory* (Wiley, 2000), response-theory chapter.
- P. C. Hariharan, J. A. Pople, *Theor. Chim. Acta* **28**, 213 (1973) —
  the 6-31G* basis.

## What is checked

`cpu::scf::RHF{basis = Pople631Gs}.hessian(frame)` →
`vib::MassWeighting` + `vib::HessianDiagonalizer` gives 3 real modes
(`n_imaginary == 0` is NOT asserted — this geometry is slightly off the
HF/6-31G* minimum, but all three vibrational modes are real and positive);
each mode within 1 cm⁻¹ of (a) the `cpu::vib::Hessian` FD production line
with the same RHF, and (b) the pinned PySCF values. Failure prints
mode-by-mode got / want / tol and exits non-zero. ≤ 60 s, single node.
