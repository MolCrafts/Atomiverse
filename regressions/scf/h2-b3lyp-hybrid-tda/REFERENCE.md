# H₂ B3LYP/STO-3G hybrid TDDFT-TDA S₁ — reference values

## Method

Closed-shell **hybrid TDDFT within the Tamm–Dancoff approximation (TDA)** on a
**B3LYP** Kohn–Sham reference. Ground state is RKS/B3LYP; the linear-response
Casida **A** matrix includes the **exact-exchange kernel** contribution of the
hybrid functional (not pure ALDA). Observable: first singlet excitation energy
**S₁ [eV]** (only one singles channel on H₂/STO-3G: 1 occ × 1 virt).

This pin is the oracle twin for the missing hybrid-kernel axis on `RKS::tda`
(roadmap: hybrid TDDFT ○ after full MO-product LSDA ALDA pure TDA closed).
It is **not** a pure-functional ALDA duplicate of `h2_rks_alda_tda` /
`h2_full_grid_alda`.

## Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase (box irrelevant to isolated-molecule SCF).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

(Equivalent to placing the bond on +x with the same bond length.)

## Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells, so
  cart/sph is a no-op here but is recorded for protocol consistency)

## Convergence

- Ground-state SCF: `conv_tol = 1e-12`
- TDA: default PySCF TDA settings; `nstates = 3` requested (only **1** root
  exists: n_occ=1, n_virt=1 for closed-shell H₂/STO-3G)

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks-tda \
  --geom h2 --basis sto-3g --xc b3lyp --nstates 3 --conv-tol 1e-12
```

(`rks-tda` builds `dft.RKS(mol)` with `mf.xc = 'b3lyp'`, then `tdscf.TDA(mf)`.)

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

### Pinned output (2026-07-23)

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

## Context (same geom/basis; not gates)

| Theory | S₁ [eV] |
|--------|---------|
| CIS / RHF-TDA | 25.8075 |
| **B3LYP hybrid TDA** | **25.9284** |
| PBE0 hybrid TDA | 26.1374 |
| LSDA pure TDA | 26.4430 |

Hybrid B3LYP sits between pure CIS and pure LSDA TDA on this tiny system, as
expected for a partial exact-exchange kernel.

## Tolerance and rationale

**|ω_S₁ − 25.928364822972 eV| < 0.05 eV** (absolute).

Rationale: hybrid TDDFT response is a new kernel path (exact exchange in **A**);
0.05 eV is a honest ≤60 s public-API end-to-end sanity band for the first
engine implementation, not a milli-eV scientific acceptance test. Ground-state
E_RKS may be checked at a looser band (e.g. 1e-4 Ha) if the regression also
asserts SCF energy; the primary observable for this slug is **S₁ [eV]**.

## What the regression should check (when implemented)

- Public API: concrete KS method with hybrid XC (B3LYP), Frame I/O only —
  no `*Config` bag, no live PySCF.
- `RKS` (or equivalent) converges; hybrid TDA returns at least one singlet.
- S₁ [eV] within **0.05 eV** of the hard-coded pin **25.928364822972**.
- Optional: length-gauge oscillator strength within a generous band of **1.100**.

## Literature pointers (method, not numbers)

- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP hybrid form
  (PySCF `xc='B3LYP'` parameterization).
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.
