# CH2O HF/6-31G* TDA + TDHF/RPA excitation energies — reference values

## Literature basis

- S. Hirata, M. Head-Gordon, *Time-dependent density functional theory
  within the Tamm-Dancoff approximation*, Chem. Phys. Lett. **314**, 291
  (1999). DOI 10.1016/S0009-2614(99)01149-5 — the TDA: drop B, solve the
  Hermitian A X = ωX (on an HF reference this is CIS); real spectrum on any
  reference.
- A. Dreuw, M. Head-Gordon, *Single-reference ab initio methods for the
  calculation of excited states of large molecules*, Chem. Rev. **105**,
  4009 (2005). DOI 10.1021/cr0505627 — TDHF/RPA response formalism
  [[A,B],[−B,−A]](X;Y) = ω(X;Y), length-gauge transition moments and
  oscillator strengths, triplet-instability pathology of full RPA.
- A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, §7 (TDHF/RPA and
  CIS; the ω_RPA ≤ ω_TDA variational relation per root follows from the
  Hermitian reduced form (A−B)^{1/2}(A+B)(A−B)^{1/2} Z = ω²Z).
- Formaldehyde is the classic textbook case: the lowest singlet is the dark
  n→π* (¹A₂, dipole-forbidden), and the bright π→π* (¹A₁) transition dipole
  lies along the C=O axis.

## Geometry (identical literals in reg_ch2o_tdhf.cpp)

Standard textbook geometry, planar in the xz plane, O on +z:
r(CO) = 1.203 Å, r(CH) = 1.099 Å, ∠HCH = 116.5°.

```
C 0 0 0
O 0 0 1.203
H  0.9345370953 0 -0.5783091021
H -0.9345370953 0 -0.5783091021
```

(H coordinates: x = ±1.099·sin(116.5°/2), z = −1.099·cos(116.5°/2).)

## Oracle: PySCF 2.14.0 (external run — nothing here reads PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf 2.14.0). 6-31G* is run with
`cart=True` (Pople bases = Cartesian d, engine convention).

Command (verbatim):

```bash
~/work/.x86_64/bin/python - <<'EOF'
import numpy as np, math
from pyscf import gto, scf, tdscf
th = math.radians(116.5/2.0)
hx = 1.099*math.sin(th); hz = -1.099*math.cos(th)
atom = f"C 0 0 0; O 0 0 1.203; H {hx:.10f} 0 {hz:.10f}; H {-hx:.10f} 0 {hz:.10f}"
mol = gto.M(atom=atom, basis="6-31g*", cart=True, unit="Angstrom", verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e = mf.kernel()
print("E_RHF = %.12f" % e)
for name, TD in (("TDA", tdscf.TDA), ("TDHF", tdscf.TDHF)):
    td = TD(mf); td.nstates = 8; td.conv_tol = 1e-10; td.max_cycle = 200
    td.kernel()
    f = td.oscillator_strength()
    print(name, "converged:", td.converged[:3])
    for k in range(3):
        print("  root %d: e = %.12f Ha   f = %.10f" % (k+1, td.e[k], f[k]))
EOF
```

Note: `nstates = 8` with `conv_tol = 1e-10` is used because PySCF's TDHF
Davidson skips the 9.71 eV root when asked for exactly 3 states at
`conv_tol = 1e-12` (root-homing artifact of its solver); the first-3-root
values below are identical (< 1e-12 Ha) between the `nstates = 3, 5, 8`
runs that do converge.

Output (pins):

```
E_RHF = -113.865575118441
TDA converged: [ True  True False]
  root 1: e = 0.171581500059 Ha   f = 0.0000000000
  root 2: e = 0.365672890232 Ha   f = 0.0013120301
  root 3: e = 0.378524177461 Ha   f = 0.2165661719
TDHF converged: [ True  True  True]
  root 1: e = 0.165369739863 Ha   f = 0.0000000000
  root 2: e = 0.356745006619 Ha   f = 0.0008026642
  root 3: e = 0.358521315202 Ha   f = 0.1795988961
```

(The TDA root-3 `converged: False` flag at these settings is another
Davidson bookkeeping artifact: the same three values are reproduced to
< 1e-12 Ha by the fully-converged `nstates = 5, conv_tol = 1e-9` run.)

Assignments (from `td.xy` dominant amplitudes + `td.transition_dipole()`):

| root | dominant excitation | character | f (TDA) | tdm direction |
|------|--------------------|-----------|---------|---------------|
| 1 | HOMO (n_O, b₂) → LUMO (π*, b₁) | ¹A₂ n→π* | 0 (dark) | forbidden |
| 2 | HOMO−2 (σ_CO) → LUMO (π*) | σ→π* | 0.0013 | y (weak) |
| 3 | HOMO−1 (π_CO, b₁) → LUMO (π*, b₁) | ¹A₁ π→π* | 0.2166 | z (C=O axis) |

## Units

Pins are stored in **Hartree** and converted in the regression with the
engine's own `units::kHartreeInEv` (CODATA, 1 Ha = 27.211386... eV), so the
1e-4 eV gate never mixes two different conversion constants. PySCF's
`td.e` is in Hartree natively; its oscillator strengths are the
length-gauge dimensionless f = (2/3)·ω[Ha]·|μ|²[a.u.] — the same formula
the engine implements.

## Gates

- TDA roots 1-3 and TDHF roots 1-3: |ours − pin·kHartreeInEv| < 1e-4 eV.
- ω_RPA ≤ ω_TDA per root (variational property of dropping B).
- n→π* (root 1) dark: f < 1e-3 in both theories.
- π→π* (root 3) bright: |f − pin| < 1e-3, transition dipole along z.
- E_RHF: |ours − pin| < 1e-6 Ha.
