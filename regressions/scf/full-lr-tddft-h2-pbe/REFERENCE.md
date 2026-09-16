# H₂ PBE/STO-3G full linear-response TDDFT S₁ — reference values

## Method

Closed-shell **full linear-response TDDFT (RPA / Casida A,B)** on a **PBE**
Kohn–Sham reference — **not** the Tamm–Dancoff approximation. Ground state is
RKS/PBE (pure GGA, \(\alpha_{\mathrm{hf}}=0\)); the response uses PySCF
`tdscf.TDDFT` (full RPA with adiabatic semi-local \(f^{\mathrm{xc}}\) on both
A and B), not `tdscf.TDA`. Observable: first singlet excitation energy
**S₁ [eV]** (only one singles channel on H₂/STO-3G: 1 occ × 1 virt).
Length-gauge oscillator strength is recorded optionally.

This pin is the oracle for roadmap **`full-lr-tddft-h2-pbe`** (Casida A/B
beyond TDA on pure KS). It is **not** a TDA duplicate and **not** the B3LYP
hybrid RPA pin (`h2-b3lyp-rpa-tddft`).

## Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase (box irrelevant to isolated-molecule SCF).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

## Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells, so
  cart/sph is a no-op here but is recorded for protocol consistency)

## Convergence

- Ground-state SCF: `conv_tol = 1e-12`
- TDDFT (full RPA): default PySCF `TDDFT` settings; `nstates = 3` requested
  (only **1** root exists: n_occ=1, n_virt=1 for closed-shell H₂/STO-3G)

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks-tddft \
  --geom h2 --basis sto-3g --xc pbe --nstates 3 --conv-tol 1e-12
```

(`rks-tddft` builds `dft.RKS(mol)` with `mf.xc = 'pbe'`, then
`tdscf.TDDFT(mf)` — full RPA Casida **(A,B)**, not TDA.)

### Verbatim twin of the literature-style call sequence

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, dft
HA2EV = 27.211386245988  # CODATA; Atomiverse units::kHartreeInEv
mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="Angstrom",
    cart=True,
    verbose=0,
)
mf = dft.RKS(mol, xc="PBE")
mf.conv_tol = 1e-12
e = mf.kernel()
td = mf.TDDFT()  # full RPA, not TDA
td.nstates = 3
td.kernel()
print("E_RKS = %.12f Ha  converged=%s" % (e, mf.converged))
print("S1 = %.12f Ha = %.12f eV  f=%.10f  td_conv=%s"
      % (td.e[0], td.e[0] * HA2EV, td.oscillator_strength()[0], td.converged[0]))
EOF
```

### Pinned output (2026-07-23)

```
E_RKS = -1.152072795159 Ha  converged=True
S1    =  0.937822497995 Ha = 25.519450223124 eV
f     =  0.858369934767   (length gauge)
td_conv = True
```

JSON from `pin_scf.py rks-tddft` (subset):

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `xc` | pbe |
| `e_tot_ha` | **−1.1520727951590923** |
| `excitation_ha[0]` (S₁) | **0.937822497995188** |
| `excitation_ev[0]` (S₁) | **25.519450223124366** |
| `oscillator_strength[0]` | 0.8583699347665188 |
| `td_converged[0]` | true |

Ha→eV conversion in the pin helper uses **27.211386245988** (CODATA), matching
Atomiverse `units::kHartreeInEv` so the regression gate does not mix conversion
constants.

### Contrast: same geom/basis/XC, TDA only (not a gate)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks-tda \
  --geom h2 --basis sto-3g --xc pbe --nstates 3 --conv-tol 1e-12
```

| Theory | S₁ [eV] | f (length) |
|--------|---------|------------|
| **PBE full RPA TDDFT** | **25.519450223124** | **0.858** |
| PBE TDA | 26.210883282418 | 1.112 |

Full RPA (A,B) lies **below** TDA (A-only) on this system, as expected when
de-excitation channels are retained. Ground-state E_RKS is identical for both
pins (−1.152072795159 Ha).

## Tolerance and rationale

**|ω_S₁ − 25.519450223124366 eV| < 1e-4 eV** (absolute).

Rationale: job gate for `full-lr-tddft-h2-pbe` is **1e-4 eV** on S₁. Pure-GGA
full LR-TDDFT with adiabatic \(f^{\mathrm{xc}}\) must match the PySCF pin at
this band on H₂/STO-3G (n_ov = 1). Ground-state E_RKS may be checked at a
looser band (e.g. 1e-4 Ha) if the regression also asserts SCF energy; the
primary observable is **S₁ [eV]**.

## What the regression should check

- Public API: concrete `RKS` with `XCType::PBE`, Frame I/O only — no `*Config`
  bag, no live PySCF.
- `RKS` converges; **`RKS::tddft`** (not `tda`) returns at least one singlet
  with `tda == false`.
- S₁ [eV] within **1e-4 eV** of the hard-coded pin **25.519450223124366**.
- Optional: length-gauge oscillator strength within a generous band of **0.858**.
- Must **not** hard-gate against the TDA pin (26.211 eV) — different theory.

## Literature pointers (method, not numbers)

- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995) — Casida equations (A,B).
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs full.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.
