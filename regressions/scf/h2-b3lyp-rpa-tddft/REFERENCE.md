# H₂ B3LYP/STO-3G full RPA TDDFT S₁ — reference values

## Method

Closed-shell **full linear-response TDDFT (RPA / Casida A,B)** on a **B3LYP**
Kohn–Sham reference — **not** the Tamm–Dancoff approximation. Ground state is
RKS/B3LYP; the response uses `tdscf.TDDFT` (PySCF’s full RPA TDDFT with hybrid
kernel / exact-exchange contribution), not `tdscf.TDA`. Observable: first
singlet excitation energy **S₁ [eV]** (only one singles channel on H₂/STO-3G:
1 occ × 1 virt). Length-gauge oscillator strength is recorded optionally.

This pin is the oracle twin for roadmap **full linear-response TDDFT** on
`RKS` (CPU ○). It is **not** a TDA duplicate of `h2-b3lyp-hybrid-tda` (same
geom/basis/XC, different response: A,B vs A-only).

## Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase (box irrelevant to isolated-molecule SCF).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

(Same geometry as `h2-b3lyp-hybrid-tda`.)

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
  --geom h2 --basis sto-3g --xc b3lyp --nstates 3 --conv-tol 1e-12
```

(`rks-tddft` builds `dft.RKS(mol)` with `mf.xc = 'b3lyp'`, then
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

### Pinned output (2026-07-23)

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

## Context (same geom/basis/XC; contrast, not gates)

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

## Tolerance and rationale

**|ω_S₁ − 25.30779734533265 eV| < 1e-4 eV** (absolute).

Rationale: job gate for `h2-b3lyp-rpa-tddft` is **1e-4 eV** on S₁. Hybrid
full LR-TDDFT (Casida A/B + α_hf + adiabatic f^xc) must match the PySCF pin
at this band on H₂/STO-3G (n_ov = 1). Ground-state E_RKS may be checked at
1e-4 Ha; the primary observable is **S₁ [eV]**.

## What the regression checks

- Public API: concrete `RKS` with `XCType::B3LYP`, Frame I/O only — no
  `*Config` bag, no live PySCF.
- `RKS` converges; **`RKS::tddft`** (not `tda`) returns at least one singlet
  with `tda == false` and `alda == false`.
- S₁ [eV] within **1e-4 eV** of the hard-coded pin **25.30779734533265**.
- Optional: length-gauge oscillator strength within a generous band of **0.861**.
- Must **not** hard-gate against the TDA pin (25.928 eV) — different theory.

## Performance (fair bench vs PySCF)

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

## Literature pointers (method, not numbers)

- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995) — Casida equations (A,B).
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs full.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP hybrid form
  (PySCF `xc='B3LYP'` parameterization).
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.
