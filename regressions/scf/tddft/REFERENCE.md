# scf/tddft

One case; every section below is run by `atv_regressions --case scf/tddft` and keeps its own oracle record.


## CH2O HF/6-31G* TDA + TDHF/RPA excitation energies — reference values

### Literature basis

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

### Geometry (identical literals in reg_ch2o_tdhf.cpp)

Standard textbook geometry, planar in the xz plane, O on +z:
r(CO) = 1.203 Å, r(CH) = 1.099 Å, ∠HCH = 116.5°.

```
C 0 0 0
O 0 0 1.203
H  0.9345370953 0 -0.5783091021
H -0.9345370953 0 -0.5783091021
```

(H coordinates: x = ±1.099·sin(116.5°/2), z = −1.099·cos(116.5°/2).)

### Oracle: PySCF 2.14.0 (external run — nothing here reads PySCF)

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

### Units

Pins are stored in **Hartree** and converted in the regression with the
engine's own `units::kHartreeInEv` (CODATA, 1 Ha = 27.211386... eV), so the
1e-4 eV gate never mixes two different conversion constants. PySCF's
`td.e` is in Hartree natively; its oscillator strengths are the
length-gauge dimensionless f = (2/3)·ω[Ha]·|μ|²[a.u.] — the same formula
the engine implements.

### Gates

- TDA roots 1-3 and TDHF roots 1-3: |ours − pin·kHartreeInEv| < 1e-4 eV.
- ω_RPA ≤ ω_TDA per root (variational property of dropping B).
- n→π* (root 1) dark: f < 1e-3 in both theories.
- π→π* (root 3) bright: |f − pin| < 1e-3, transition dipole along z.
- E_RHF: |ours − pin| < 1e-6 Ha.


## H₂ B3LYP/STO-3G hybrid TDDFT-TDA S₁ — reference values

### Method

Closed-shell **hybrid TDDFT within the Tamm–Dancoff approximation (TDA)** on a
**B3LYP** Kohn–Sham reference. Ground state is RKS/B3LYP; the linear-response
Casida **A** matrix includes the **exact-exchange kernel** contribution of the
hybrid functional (not pure ALDA). Observable: first singlet excitation energy
**S₁ [eV]** (only one singles channel on H₂/STO-3G: 1 occ × 1 virt).

This pin is the oracle twin for the missing hybrid-kernel axis on `RKS::tda`
(roadmap: hybrid TDDFT ○ after full MO-product LSDA ALDA pure TDA closed).
It is **not** a pure-functional ALDA duplicate of `h2_full_grid_alda`.

### Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase (box irrelevant to isolated-molecule SCF).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

(Equivalent to placing the bond on +x with the same bond length.)

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells, so
  cart/sph is a no-op here but is recorded for protocol consistency)

### Convergence

- Ground-state SCF: `conv_tol = 1e-12`
- TDA: default PySCF TDA settings; `nstates = 3` requested (only **1** root
  exists: n_occ=1, n_virt=1 for closed-shell H₂/STO-3G)

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks-tda \
  --geom h2 --basis sto-3g --xc b3lyp --nstates 3 --conv-tol 1e-12
```

(`rks-tda` builds `dft.RKS(mol)` with `mf.xc = 'b3lyp'`, then `tdscf.TDA(mf)`.)

#### Verbatim twin of the literature-style call sequence

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
mf = dft.RKS(mol, xc="B3LYP")
mf.conv_tol = 1e-12
e = mf.kernel()
td = mf.TDA()
td.nstates = 3
td.kernel()
print("E_RKS = %.12f Ha  converged=%s" % (e, mf.converged))
print("S1 = %.12f Ha = %.12f eV  f=%.10f  td_conv=%s"
      % (td.e[0], td.e[0] * HA2EV, td.oscillator_strength()[0], td.converged[0]))
EOF
```

#### Pinned output (2026-07-23)

```
E_RKS = -1.165418410673 Ha  converged=True
S1    =  0.952849832367 Ha = 25.928364822972 eV
f     =  1.100141468478   (length gauge)
td_conv = True
```

JSON from `pin_scf.py rks-tda` (subset):

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `xc` | b3lyp |
| `e_tot_ha` | **−1.1654184106731211** |
| `excitation_ha[0]` (S₁) | **0.9528498323673198** |
| `excitation_ev[0]` (S₁) | **25.928364822972057** |
| `oscillator_strength[0]` | 1.1001414684779438 |
| `td_converged[0]` | true |

Ha→eV conversion in the pin helper uses **27.211386245988** (CODATA), matching
Atomiverse `units::kHartreeInEv` so the regression gate does not mix conversion
constants.

### Context (same geom/basis; not gates)

| Theory | S₁ [eV] |
|--------|---------|
| CIS / RHF-TDA | 25.8075 |
| **B3LYP hybrid TDA** | **25.9284** |
| PBE0 hybrid TDA | 26.1374 |
| LSDA pure TDA | 26.4430 |

Hybrid B3LYP sits between pure CIS and pure LSDA TDA on this tiny system, as
expected for a partial exact-exchange kernel.

### Tolerance and rationale

**|ω_S₁ − 25.928364822972 eV| < 0.05 eV** (absolute).

Rationale: hybrid TDDFT response is a new kernel path (exact exchange in **A**);
0.05 eV is a honest ≤60 s public-API end-to-end sanity band for the first
engine implementation, not a milli-eV scientific acceptance test. Ground-state
E_RKS may be checked at a looser band (e.g. 1e-4 Ha) if the regression also
asserts SCF energy; the primary observable for this slug is **S₁ [eV]**.

### What the regression should check (when implemented)

- Public API: concrete KS method with hybrid XC (B3LYP), Frame I/O only —
  no `*Config` bag, no live PySCF.
- `RKS` (or equivalent) converges; hybrid TDA returns at least one singlet.
- S₁ [eV] within **0.05 eV** of the hard-coded pin **25.928364822972**.
- Optional: length-gauge oscillator strength within a generous band of **1.100**.

### Literature pointers (method, not numbers)

- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP hybrid form
  (PySCF `xc='B3LYP'` parameterization).
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.


## H₂ B3LYP/STO-3G full RPA TDDFT S₁ — reference values

### Method

Closed-shell **full linear-response TDDFT (RPA / Casida A,B)** on a **B3LYP**
Kohn–Sham reference — **not** the Tamm–Dancoff approximation. Ground state is
RKS/B3LYP; the response uses `tdscf.TDDFT` (PySCF’s full RPA TDDFT with hybrid
kernel / exact-exchange contribution), not `tdscf.TDA`. Observable: first
singlet excitation energy **S₁ [eV]** (only one singles channel on H₂/STO-3G:
1 occ × 1 virt). Length-gauge oscillator strength is recorded optionally.

This pin is the oracle twin for roadmap **full linear-response TDDFT** on
`RKS` (CPU ○). It is **not** a TDA duplicate of `h2-b3lyp-hybrid-tda` (same
geom/basis/XC, different response: A,B vs A-only).

### Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase (box irrelevant to isolated-molecule SCF).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

(Same geometry as `h2-b3lyp-hybrid-tda`.)

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells, so
  cart/sph is a no-op here but is recorded for protocol consistency)

### Convergence

- Ground-state SCF: `conv_tol = 1e-12`
- TDDFT (full RPA): default PySCF `TDDFT` settings; `nstates = 3` requested
  (only **1** root exists: n_occ=1, n_virt=1 for closed-shell H₂/STO-3G)

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks-tddft \
  --geom h2 --basis sto-3g --xc b3lyp --nstates 3 --conv-tol 1e-12
```

(`rks-tddft` builds `dft.RKS(mol)` with `mf.xc = 'b3lyp'`, then
`tdscf.TDDFT(mf)` — full RPA Casida **(A,B)**, not TDA.)

#### Verbatim twin of the literature-style call sequence

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
mf = dft.RKS(mol, xc="B3LYP")
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

#### Pinned output (2026-07-23)

```
E_RKS = -1.165418410673 Ha  converged=True
S1    =  0.930044398200 Ha = 25.307797345333 eV
f     =  0.860889120813   (length gauge)
td_conv = True
```

JSON from `pin_scf.py rks-tddft` (subset):

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `xc` | b3lyp |
| `e_tot_ha` | **−1.1654184106731211** |
| `excitation_ha[0]` (S₁) | **0.9300443981998157** |
| `excitation_ev[0]` (S₁) | **25.30779734533265** |
| `oscillator_strength[0]` | 0.8608891208128704 |
| `td_converged[0]` | true |

Ha→eV conversion in the pin helper uses **27.211386245988** (CODATA), matching
Atomiverse `units::kHartreeInEv` so the regression gate does not mix conversion
constants.

### Context (same geom/basis/XC; contrast, not gates)

| Theory | S₁ [eV] | f (length) |
|--------|---------|------------|
| CIS / RHF-TDA | 25.8075 | — |
| **B3LYP hybrid full RPA TDDFT** | **25.3078** | **0.861** |
| B3LYP hybrid TDA | 25.9284 | 1.100 |
| PBE0 hybrid TDA | 26.1374 | — |
| LSDA pure TDA | 26.4430 | — |

Full RPA (A,B) lies **below** hybrid TDA (A-only) on this system, as expected
when de-excitation channels are retained. Ground-state E_RKS is identical for
TDA and RPA pins (−1.165418410673 Ha).

### Tolerance and rationale

**|ω_S₁ − 25.30779734533265 eV| < 1e-4 eV** (absolute).

Rationale: job gate for `h2-b3lyp-rpa-tddft` is **1e-4 eV** on S₁. Hybrid
full LR-TDDFT (Casida A/B + α_hf + adiabatic f^xc) must match the PySCF pin
at this band on H₂/STO-3G (n_ov = 1). Ground-state E_RKS may be checked at
1e-4 Ha; the primary observable is **S₁ [eV]**.

### What the regression checks

- Public API: concrete `RKS` with `XCType::B3LYP`, Frame I/O only — no
  `*Config` bag, no live PySCF.
- `RKS` converges; **`RKS::tddft`** (not `tda`) returns at least one singlet
  with `tda == false` and `alda == false`.
- S₁ [eV] within **1e-4 eV** of the hard-coded pin **25.30779734533265**.
- Optional: length-gauge oscillator strength within a generous band of **0.861**.
- Must **not** hard-gate against the TDA pin (25.928 eV) — different theory.

### Performance (fair bench vs PySCF)

Same geom/basis/method class: RKS/B3LYP + full RPA TDDFT (`rks_tddft`).
Median of ≥3 runs; ratio = T_atv / T_pyscf; target ≤ 10.0.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method rks_tddft --geom h2 --basis sto-3g --xc b3lyp --nstates 3 \
    --repeat 3 --max-ratio 10.0 -- \
    ./build/regressions/reg_h2-b3lyp-rpa-tddft
'
```

| Field | Value |
|-------|-------|
| `ratio_atv_over_pyscf` | **0.430** |
| `within_10x` | **true** (≤ 10.0) |
| `pyscf_median_s` | 0.172 |
| `atv_median_s` | 0.074 |
| measured | 2026-07-23; `OMP_NUM_THREADS=1` both sides |

### Literature pointers (method, not numbers)

- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995) — Casida equations (A,B).
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs full.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP hybrid form
  (PySCF `xc='B3LYP'` parameterization).
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.


## H₂ PBE/STO-3G full linear-response TDDFT S₁ — reference values

### Method

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

### Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase (box irrelevant to isolated-molecule SCF).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells, so
  cart/sph is a no-op here but is recorded for protocol consistency)

### Convergence

- Ground-state SCF: `conv_tol = 1e-12`
- TDDFT (full RPA): default PySCF `TDDFT` settings; `nstates = 3` requested
  (only **1** root exists: n_occ=1, n_virt=1 for closed-shell H₂/STO-3G)

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks-tddft \
  --geom h2 --basis sto-3g --xc pbe --nstates 3 --conv-tol 1e-12
```

(`rks-tddft` builds `dft.RKS(mol)` with `mf.xc = 'pbe'`, then
`tdscf.TDDFT(mf)` — full RPA Casida **(A,B)**, not TDA.)

#### Verbatim twin of the literature-style call sequence

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

#### Pinned output (2026-07-23)

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

#### Contrast: same geom/basis/XC, TDA only (not a gate)

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

### Tolerance and rationale

**|ω_S₁ − 25.519450223124366 eV| < 1e-4 eV** (absolute).

Rationale: job gate for `full-lr-tddft-h2-pbe` is **1e-4 eV** on S₁. Pure-GGA
full LR-TDDFT with adiabatic \(f^{\mathrm{xc}}\) must match the PySCF pin at
this band on H₂/STO-3G (n_ov = 1). Ground-state E_RKS may be checked at a
looser band (e.g. 1e-4 Ha) if the regression also asserts SCF energy; the
primary observable is **S₁ [eV]**.

### What the regression should check

- Public API: concrete `RKS` with `XCType::PBE`, Frame I/O only — no `*Config`
  bag, no live PySCF.
- `RKS` converges; **`RKS::tddft`** (not `tda`) returns at least one singlet
  with `tda == false`.
- S₁ [eV] within **1e-4 eV** of the hard-coded pin **25.519450223124366**.
- Optional: length-gauge oscillator strength within a generous band of **0.858**.
- Must **not** hard-gate against the TDA pin (26.211 eV) — different theory.

### Literature pointers (method, not numbers)

- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995) — Casida equations (A,B).
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs full.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.


## H2 full-grid ALDA TDA

### System

H₂ at r = 0.74 Å, STO-3G, RKS-LSDA, closed-shell TDA singlet, full
MO-product ALDA \(f^{\mathrm{xc}}\) on the SG0 (50×110) grid.

### Oracle

Atomiverse `RKS::tda` full-grid ALDA path (Dirac-exchange second derivative),
2026-07-23. Not a third-party live oracle — value is hard-coded in the
regression binary after calibration against this engine.

### Gate

- `es.alda == true`
- S1 excitation energy [eV] within documented absolute tolerance


## O₂ UKS-LSDA/STO-3G spin-conserving TDA — S/T-manifold roots [eV]

### Method

Open-shell **Tamm–Dancoff TDDFT** on a **UKS/LSDA** reference (triplet O₂,
\(2S=2\)). Ground state is unrestricted Kohn–Sham with Slater exchange + VWN5
correlation (Atomiverse `XCType::LSDA` ≡ PySCF `xc="LDA,VWN"`). Linear response
is spin-conserving Casida **A** (no closed-shell singlet/triplet channel split):

\[
A_{ia\sigma,jb\sigma'}
=
\delta_{\sigma\sigma'}\delta_{ij}\delta_{ab}
(\varepsilon_a^\sigma-\varepsilon_i^\sigma)
+(ia\sigma|jb\sigma')
+(ia\sigma|f^{\mathrm{xc}}_{\sigma\sigma'}|jb\sigma')
\]

with pure DFT \(\alpha_{\mathrm{hf}}=0\) (no exact-exchange subtraction) and
adiabatic spin-resolved \(f^{\mathrm{xc}}\) from the LSDA second derivatives on
the KS grid (PySCF `tdscf.TDA` on `dft.UKS`).

Observable: lowest spin-conserving excitation roots **ω [eV]** (Ms-preserving
excitations of the triplet reference — the open-shell analogue of the S/T
root list on closed-shell TDA).

### Geometry

O₂, r(O–O) = **1.2075 Å**, charge 0, spin \(2S=2\) (triplet).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.0   | 0.0   | 0.0   |
| O    | 0.0   | 0.0   | 1.2075 |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True`; STO-3G has no \(d\), cart flag is inert
  but recorded for protocol consistency)
- Occupation: \((n_\alpha, n_\beta) = (9, 7)\)

### Convergence

- Ground-state SCF: `conv_tol = 1e-12`
- TDA: default PySCF TDA; `nstates = 5`

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py uks-tda \
  --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --nstates 5 --conv-tol 1e-12
```

#### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, dft, tdscf
HA2EV = 27.211386245988  # CODATA; Atomiverse units::kHartreeInEv
mol = gto.M(
    atom="O 0 0 0; O 0 0 1.2075",
    basis="sto-3g",
    spin=2,
    unit="Angstrom",
    cart=True,
    verbose=0,
)
mf = dft.UKS(mol)
mf.xc = "LDA,VWN"
mf.conv_tol = 1e-12
e = mf.kernel()
td = tdscf.TDA(mf)
td.nstates = 5
td.kernel()
print("E_UKS = %.12f Ha  S2=%.10f  converged=%s" % (e, mf.spin_square()[0], mf.converged))
for i, w in enumerate(td.e):
    print("root %d  %.12f Ha = %.12f eV  f=%.6e  conv=%s"
          % (i, w, w * HA2EV, td.oscillator_strength()[i], td.converged[i]))
EOF
```

#### Pinned output (2026-07-23)

```
E_UKS = -147.193805922283 Ha  S2≈2.000202  converged=True
root 0  0.280063546008 Ha =  7.620917323839 eV
root 1  0.280854853467 Ha =  7.642449896744 eV
root 2  0.280854872276 Ha =  7.642450408571 eV
root 3  0.308815024999 Ha =  8.403284923812 eV
root 4  0.308815024999 Ha =  8.403284923812 eV
```

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `xc` | LDA,VWN |
| `e_tot_ha` | **−147.193805922283** |
| `excitation_ev[0]` | **7.620917323839** |
| `excitation_ev[1]` | **7.642449896744** |
| `excitation_ev[2]` | **7.642450408571** |
| `excitation_ev[3]` | **8.403284923812** |
| `excitation_ev[4]` | **8.403284923812** |

Ha→eV conversion uses **27.211386245988** (CODATA), matching
`units::kHartreeInEv`.

Roots 1–2 and 3–4 are near-degenerate pairs (π-manifold structure of O₂).

### Tolerance and rationale

**|ω_i − pin_i| < 1e-4 eV** (absolute) for the five lowest roots.

Rationale: open-shell UKS-TDA is a new product path (unrestricted Casida **A**
+ spin-resolved ALDA); 1e-4 eV is the job gate for this textbook STO-3G case
and is tight enough to catch kernel-sign / channel-block errors while remaining
honest for grid / SCF residual noise. Ground-state E_UKS may be checked at
~1e-4 Ha as a sanity band; the primary observables are the **excitation
energies [eV]**.

### Literature pointers (method, not numbers)

- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.


## O₂ UKS-LSDA/STO-3G full linear-response TDDFT — roots [eV]

### Method

Open-shell **full linear-response TDDFT (Casida A/B RPA)** on a **UKS/LSDA**
reference (triplet O₂, \(2S=2\)). Ground state is unrestricted Kohn–Sham with
Slater exchange + VWN5 correlation (Atomiverse `XCType::LSDA` ≡ PySCF
`xc="LDA,VWN"`). Linear response is spin-conserving Casida **A** and **B**
(not TDA):

\[
A_{ia\sigma,jb\sigma'}
=
\delta_{\sigma\sigma'}\delta_{ij}\delta_{ab}
(\varepsilon_a^\sigma-\varepsilon_i^\sigma)
+(ia\sigma|jb\sigma')
+(ia\sigma|f^{\mathrm{xc}}_{\sigma\sigma'}|jb\sigma')
\]

\[
B_{ia\sigma,jb\sigma'}
=
(ia\sigma|bj\sigma')
+(ia\sigma|f^{\mathrm{xc}}_{\sigma\sigma'}|jb\sigma')
\]

with pure DFT \(\alpha_{\mathrm{hf}}=0\) and adiabatic spin-resolved
\(f^{\mathrm{xc}}\) from the LSDA second derivatives on the KS grid (PySCF
`tdscf.TDDFT` on `dft.UKS`). For real MOs the Coulomb/fxc blocks of **A** and
**B** coincide, so \(A-B=\mathrm{diag}(\varepsilon_a-\varepsilon_i)\).

Observable: lowest spin-conserving excitation roots **ω [eV]** (Ms-preserving
excitations of the triplet reference).

This pin is **not** a TDA duplicate (`uks-tda-o2-lsda`); full RPA retains
de-excitation channels and shifts the higher roots.

### Geometry

O₂, r(O–O) = **1.2075 Å**, charge 0, spin \(2S=2\) (triplet).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.0   | 0.0   | 0.0   |
| O    | 0.0   | 0.0   | 1.2075 |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True`; STO-3G has no \(d\), cart flag is inert
  but recorded for protocol consistency)
- Occupation: \((n_\alpha, n_\beta) = (9, 7)\)

### Convergence

- Ground-state SCF: `conv_tol = 1e-12`
- TDDFT (full RPA): default PySCF `TDDFT`; `nstates = 5`

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py uks-tddft \
  --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --nstates 5 --conv-tol 1e-12
```

#### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, dft, tdscf
HA2EV = 27.211386245988  # CODATA; Atomiverse units::kHartreeInEv
mol = gto.M(
    atom="O 0 0 0; O 0 0 1.2075",
    basis="sto-3g",
    spin=2,
    unit="Angstrom",
    cart=True,
    verbose=0,
)
mf = dft.UKS(mol)
mf.xc = "LDA,VWN"
mf.conv_tol = 1e-12
e = mf.kernel()
td = tdscf.TDDFT(mf)  # full RPA, not TDA
td.nstates = 5
td.kernel()
print("E_UKS = %.12f Ha  S2=%.10f  converged=%s" % (e, mf.spin_square()[0], mf.converged))
for i, w in enumerate(td.e):
    print("root %d  %.12f Ha = %.12f eV  f=%.6e  conv=%s"
          % (i, w, w * HA2EV, td.oscillator_strength()[i], td.converged[i]))
EOF
```

#### Pinned output (2026-07-23)

```
E_UKS = -147.193805922283 Ha  S2≈2.000202  converged=True
root 0  0.280063546008 Ha =  7.620917323839 eV
root 1  0.280853738712 Ha =  7.642419562717 eV
root 2  0.280853757468 Ha =  7.642420073101 eV
root 3  0.308191621421 Ha =  8.386321248270 eV
root 4  0.308191621421 Ha =  8.386321248273 eV
```

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `xc` | LDA,VWN |
| `e_tot_ha` | **−147.193805922283** |
| `excitation_ev[0]` | **7.620917323839** |
| `excitation_ev[1]` | **7.642419562717** |
| `excitation_ev[2]` | **7.642420073101** |
| `excitation_ev[3]` | **8.386321248270** |
| `excitation_ev[4]` | **8.386321248273** |

Ha→eV conversion uses **27.211386245988** (CODATA), matching
`units::kHartreeInEv`.

#### Contrast: same geom/basis/XC, TDA only (not a gate)

| Theory | root 0 [eV] | root 3 [eV] |
|--------|-------------|-------------|
| **UKS full RPA TDDFT** | **7.620917** | **8.386321** |
| UKS TDA | 7.620917 | 8.403285 |

Full RPA leaves the lowest root essentially unchanged and **lowers** roots 3–4
by ~0.017 eV relative to TDA — the regression must not hard-gate against the
TDA pin.

### Tolerance and rationale

**|ω_i − pin_i| < 1e-4 eV** (absolute) for the five lowest roots.

Rationale: open-shell UKS full-LR is a new product path (unrestricted Casida
**A/B** + spin-resolved ALDA); 1e-4 eV is the job gate for this textbook STO-3G
case and is tight enough to catch kernel-sign / B-matrix / channel-block errors
while remaining honest for grid / SCF residual noise. Ground-state E_UKS may be
checked at ~1e-4 Ha as a sanity band; the primary observables are the
**excitation energies [eV]**.

### What the regression should check

- Public API: concrete `UKS` with `XCType::LSDA`, Frame I/O only — no `*Config`
  bag, no live PySCF.
- `UKS` converges (ROKS warm-start optional, same multi-minima caution as
  `uks-tda-o2-lsda`); **`UKS::tddft`** (not `tda`) returns ≥5 roots with
  `tda == false` and `alda == true`.
- Each root [eV] within **1e-4 eV** of the hard-coded pins above.
- Must **not** hard-gate against the TDA root-3 pin (8.403 eV).

### Fair wall twin (offline; not ctest)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method uks_tddft --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN \
  --nstates 5 --repeat 3 -- \
  ./build/regressions/reg_uks-full-lr-o2-lsda
```

| Field | Value |
|-------|-------|
| `ratio_atv_over_pyscf` | **2.937** (median walls: ATV 0.672 s / PySCF 0.229 s; `OMP_NUM_THREADS=1`, repeat 3) |
| `within_10x` | **true** |
| `perf_exempt` | false |

### Literature pointers (method, not numbers)

- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995) — Casida equations (A,B).
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs full.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.


## O₂ UHF/STO-3G collinear spin-flip full TDDFT (SF-RPA) — excitation roots [eV]

### Method

**Collinear spin-flip full linear response** (SF-TDHF / SF-RPA) on a high-spin
**UHF** reference (triplet O₂, \(2S=2\)). Ms-decreasing singles
\(i_\alpha \to a_\beta\) (majority→minority) form the response space — same
manifold as SF-TDA (`UHF::sf_tda`). With the collinear HF kernel the Coulomb
term vanishes for pure spin-flip and only exchange remains (Shao, Head-Gordon
& Krylov, *J. Chem. Phys.* **118**, 4807 (2003)):

\[
A_{ia,jb}
=
\delta_{ij}\delta_{ab}
(\varepsilon_a^\beta-\varepsilon_i^\alpha)
-(ij|ab)
\]

\[
B_{ia,jb}
=
-(ib|aj)
\]

Non-Hermitian RPA eigenproblem

\[
\begin{bmatrix}A&B\\-B&-A\end{bmatrix}
\begin{bmatrix}X\\Y\end{bmatrix}
=
\omega
\begin{bmatrix}X\\Y\end{bmatrix}
\]

solved densely (`dgeev`). On this O₂/STO-3G UHF reference \((A-B)\) is
**not** positive definite (lowest eigenvalue ≈ −0.46 Ha), so the Hermitian
Casida reduction used by closed-shell `RKS::tddft` is undefined; the product
path is the non-Hermitian form. Observable: lowest **positive real** SF-RPA
roots **ω [eV]**.

Public path: `UHF::sf_tddft(n_states)` after a converged `UHF::run`.

PySCF **mainline has no product SF-TDDFT**; the offline pin builds the same
\(A,B\) with PySCF UHF MOs + `ao2mo` (`tools/pyscf_oracle/pin_scf.py sf-tddft`).
That script is also the fair-bench twin (`--method sf_tddft`).

### Geometry

O₂, r(O–O) = **1.2075 Å**, charge 0, spin \(2S=2\) (triplet).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.0   | 0.0   | 0.0   |
| O    | 0.0   | 0.0   | 1.2075 |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True`; STO-3G has no \(d\), cart flag inert)
- Occupation: \((n_\alpha, n_\beta) = (9, 7)\); SF space \(n_{ov}=9\times3=27\)

### Convergence

- Ground-state UHF: `conv_tol = 1e-12`
- SF-RPA: dense non-Hermitian `dgeev` of the \(2n_{ov}\) RPA matrix; `nstates = 5`

### Oracle: PySCF 2.14.0 integrals (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py sf-tddft \
  --geom o2 --basis sto-3g --spin 2 --nstates 5 --conv-tol 1e-12
```

#### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, scf, ao2mo
import numpy as np
HA2EV = 27.211386245988
mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=2,
            unit="Angstrom", cart=True, verbose=0)
mf = scf.UHF(mol); mf.conv_tol = 1e-12; e = mf.kernel()
mo_a, mo_b = mf.mo_coeff; ea, eb = mf.mo_energy; occa, occb = mf.mo_occ
oa = np.where(occa>0)[0]; vb = np.where(occb==0)[0]
no, nv = len(oa), len(vb); nov = no*nv
Co, Cv = mo_a[:, oa], mo_b[:, vb]
ijab = ao2mo.general(mol, (Co, Co, Cv, Cv), compact=False).reshape(no, no, nv, nv)
ibaj = ao2mo.general(mol, (Co, Cv, Cv, Co), compact=False).reshape(no, nv, nv, no)
A = np.zeros((nov, nov)); B = np.zeros((nov, nov))
for i in range(no):
    for a in range(nv):
        p = i*nv + a
        A[p, p] = eb[vb[a]] - ea[oa[i]]
        for j in range(no):
            for b in range(nv):
                q = j*nv + b
                A[p, q] -= ijab[i, j, a, b]
                B[p, q] -= ibaj[i, b, a, j]
A = 0.5*(A+A.T); B = 0.5*(B+B.T)
M = np.block([[A, B], [-B, -A]])
w = np.linalg.eigvals(M)
pos = sorted([float(z.real) for z in w if z.real > 1e-8 and abs(z.imag) <= 1e-8])
print("E_UHF = %.12f Ha  S2=%.10f  A-B_min=%.6f" % (e, mf.spin_square()[0],
      np.linalg.eigvalsh(A-B)[0]))
for i, wi in enumerate(pos[:5]):
    print("root %d  %.12f Ha = %.12f eV" % (i, wi, wi * HA2EV))
EOF
```

#### Pinned output (2026-07-23)

```
E_UHF = -147.633946785450 Ha  S2≈2.003411  converged=True
sf_channel = alpha_to_beta  n_ov = 27  a_minus_b_min ≈ -0.461612
root 0  0.159296979793 Ha =  4.334691644974 eV
root 1  0.262955422319 Ha =  7.155381562193 eV
root 2  0.262955422319 Ha =  7.155381562193 eV
root 3  0.300659727783 Ha =  8.181367981325 eV
root 4  0.300659727783 Ha =  8.181367981325 eV
```

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `e_tot_ha` (UHF) | **−147.633946785450** |
| `a_minus_b_min` [Ha] | **−0.461612348195** |
| `excitation_ev[0]` | **4.334691644974** |
| `excitation_ev[1]` | **7.155381562193** |
| `excitation_ev[2]` | **7.155381562193** |
| `excitation_ev[3]` | **8.181367981325** |
| `excitation_ev[4]` | **8.181367981325** |

Ha→eV uses **27.211386245988** (CODATA), matching `units::kHartreeInEv`.

Compared with SF-TDA on the same reference (roots 0.045 / 1.568 / 1.568 /
3.038 / 7.849 eV), full SF-RPA shifts the spectrum: the near-zero multiplet
reconstruction and low TDA roots are no longer the lowest positive RPA poles
once the \(B\) coupling is restored under an indefinite \((A-B)\). Degenerate
pairs remain (roots 1–2 and 3–4).

### Tolerance and rationale

**|ω_i − pin_i| < 1e-4 eV** (absolute) for the five lowest positive SF-RPA roots.

Rationale: job gate for this textbook STO-3G case; tight enough to catch
exchange-sign / \(B\)-index / non-Hermitian assembly errors while remaining
honest for residual SCF / ERI / `dgeev` noise. Ground-state E_UHF may be
checked at ~1e-4 Ha as sanity; primary observables are the **excitation
energies [eV]**.

### Literature pointers (method, not numbers)

- Y. Shao, M. Head-Gordon, A. I. Krylov, *J. Chem. Phys.* **118**, 4807 (2003) — SF-TDDFT/CIS.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDHF/RPA response review.
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA (SF-TDA sibling).

### Performance

Fair wall-time vs the offline collinear SF-RPA twin (ROHF warm-start → UHF +
dense SF \(A,B\) non-Hermitian RPA; same geom/basis/method class as
`reg_sf-tddft-o2`):

```bash
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method sf_tddft --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 5 \
  -- ./build/regressions/reg_sf-tddft-o2
```

| Side | Median wall [s] (n=5) |
|------|------------------------|
| PySCF twin (ROHF→UHF + SF-RPA) | **0.0412** |
| Atomiverse `reg_sf-tddft-o2` | **0.392** |
| **ratio** \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **9.51** (≤ 10.0 ✓) |


## O₂ UHF/STO-3G collinear spin-flip TDA (SF-CIS) — excitation roots [eV]

### Method

**Collinear spin-flip Tamm–Dancoff** (SF-CIS / SF-TDA) on a high-spin **UHF**
reference (triplet O₂, \(2S=2\)). Ms-decreasing singles \(i_\alpha \to a_\beta\)
(majority→minority) form the response space. With the collinear HF kernel the
Coulomb term vanishes for pure spin-flip and only exchange remains
(Shao, Head-Gordon & Krylov, *J. Chem. Phys.* **118**, 4807 (2003)):

\[
A_{ia,jb}
=
\delta_{ij}\delta_{ab}
(\varepsilon_a^\beta-\varepsilon_i^\alpha)
-(ij|ab)
\]

Dense Hermitian solve \(A X = \omega X\). Public path: `UHF::sf_tda(n_states)`
after a converged `UHF::run`.

PySCF **mainline has no product SF-TDA**; the offline pin builds the same
matrix with PySCF UHF MOs + `ao2mo` (`tools/pyscf_oracle/pin_scf.py sf-tda`).
That script is also the fair-bench twin (`--method sf_tda`).

Observable: lowest SF roots **ω [eV]** (includes a near-zero multiplet
component from Ms-reconstruction of the high-spin reference under mild UHF
spin contamination).

### Geometry

O₂, r(O–O) = **1.2075 Å**, charge 0, spin \(2S=2\) (triplet).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.0   | 0.0   | 0.0   |
| O    | 0.0   | 0.0   | 1.2075 |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True`; STO-3G has no \(d\), cart flag inert)
- Occupation: \((n_\alpha, n_\beta) = (9, 7)\); SF space \(n_{ov}=9\times3=27\)

### Convergence

- Ground-state UHF: `conv_tol = 1e-12`
- SF-TDA: dense `eigh` of full \(A\); `nstates = 5`

### Oracle: PySCF 2.14.0 integrals (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py sf-tda \
  --geom o2 --basis sto-3g --spin 2 --nstates 5 --conv-tol 1e-12
```

#### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, scf, ao2mo
import numpy as np
HA2EV = 27.211386245988
mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=2,
            unit="Angstrom", cart=True, verbose=0)
mf = scf.UHF(mol); mf.conv_tol = 1e-12; e = mf.kernel()
mo_a, mo_b = mf.mo_coeff; ea, eb = mf.mo_energy; occa, occb = mf.mo_occ
oa = np.where(occa>0)[0]; vb = np.where(occb==0)[0]
no, nv = len(oa), len(vb)
Co, Cv = mo_a[:, oa], mo_b[:, vb]
ijab = ao2mo.general(mol, (Co, Co, Cv, Cv), compact=False).reshape(no, no, nv, nv)
A = np.zeros((no*nv, no*nv))
for i in range(no):
    for a in range(nv):
        p = i*nv + a
        A[p, p] = eb[vb[a]] - ea[oa[i]]
        for j in range(no):
            for b in range(nv):
                A[p, j*nv + b] -= ijab[i, j, a, b]
w = np.linalg.eigvalsh(A)
print("E_UHF = %.12f Ha  S2=%.10f" % (e, mf.spin_square()[0]))
for i, wi in enumerate(w[:5]):
    print("root %d  %.12f Ha = %.12f eV" % (i, wi, wi * HA2EV))
EOF
```

#### Pinned output (2026-07-23)

```
E_UHF = -147.633946785450 Ha  S2≈2.003411  converged=True
sf_channel = alpha_to_beta  n_ov = 27
root 0  0.001659121519 Ha =  0.045146996469 eV
root 1  0.057612188548 Ha =  1.567707515056 eV
root 2  0.057612188548 Ha =  1.567707515056 eV
root 3  0.111660939718 Ha =  3.038448959259 eV
root 4  0.288431775262 Ha =  7.848628442263 eV
```

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `e_tot_ha` (UHF) | **−147.633946785450** |
| `excitation_ev[0]` | **0.045146996469** |
| `excitation_ev[1]` | **1.567707515056** |
| `excitation_ev[2]` | **1.567707515056** |
| `excitation_ev[3]` | **3.038448959259** |
| `excitation_ev[4]` | **7.848628442263** |

Ha→eV uses **27.211386245988** (CODATA), matching `units::kHartreeInEv`.

Root 0 is the near-zero Ms-reconstruction of the high-spin multiplet (UHF
spin contamination); roots 1–2 are a near-degenerate pair.

### Tolerance and rationale

**|ω_i − pin_i| < 1e-4 eV** (absolute) for the five lowest SF roots.

Rationale: job gate for this textbook STO-3G case; tight enough to catch
exchange-sign / channel-block errors while remaining honest for residual
SCF / ERI noise. Ground-state E_UHF may be checked at ~1e-4 Ha as sanity;
primary observables are the **excitation energies [eV]**.

### Literature pointers (method, not numbers)

- Y. Shao, M. Head-Gordon, A. I. Krylov, *J. Chem. Phys.* **118**, 4807 (2003) — SF-TDDFT/CIS.
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — response review.

### Performance

Fair wall-time vs the offline collinear SF-CIS twin (ROHF warm-start → UHF +
dense SF-CIS \(A\); same geom/basis/method class as `reg_spin-flip-tda-o2`):

```bash
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method sf_tda --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 5 \
  -- ./build/regressions/reg_spin-flip-tda-o2
```

| Side | Median wall [s] (n=5) |
|------|------------------------|
| PySCF twin (ROHF→UHF + SF-CIS) | **0.0326** |
| Atomiverse `reg_spin-flip-tda-o2` | **0.311** |
| **ratio** \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **9.56** (≤ 10.0 ✓) |

`within_10x = true`. Protocol: median of 5 runs; `OMP_NUM_THREADS=1`.
