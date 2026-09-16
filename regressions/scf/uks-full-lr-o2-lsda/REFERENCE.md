# O₂ UKS-LSDA/STO-3G full linear-response TDDFT — roots [eV]

## Method

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

## Geometry

O₂, r(O–O) = **1.2075 Å**, charge 0, spin \(2S=2\) (triplet).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.0   | 0.0   | 0.0   |
| O    | 0.0   | 0.0   | 1.2075 |

## Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True`; STO-3G has no \(d\), cart flag is inert
  but recorded for protocol consistency)
- Occupation: \((n_\alpha, n_\beta) = (9, 7)\)

## Convergence

- Ground-state SCF: `conv_tol = 1e-12`
- TDDFT (full RPA): default PySCF `TDDFT`; `nstates = 5`

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py uks-tddft \
  --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --nstates 5 --conv-tol 1e-12
```

### Verbatim twin

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

### Pinned output (2026-07-23)

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

### Contrast: same geom/basis/XC, TDA only (not a gate)

| Theory | root 0 [eV] | root 3 [eV] |
|--------|-------------|-------------|
| **UKS full RPA TDDFT** | **7.620917** | **8.386321** |
| UKS TDA | 7.620917 | 8.403285 |

Full RPA leaves the lowest root essentially unchanged and **lowers** roots 3–4
by ~0.017 eV relative to TDA — the regression must not hard-gate against the
TDA pin.

## Tolerance and rationale

**|ω_i − pin_i| < 1e-4 eV** (absolute) for the five lowest roots.

Rationale: open-shell UKS full-LR is a new product path (unrestricted Casida
**A/B** + spin-resolved ALDA); 1e-4 eV is the job gate for this textbook STO-3G
case and is tight enough to catch kernel-sign / B-matrix / channel-block errors
while remaining honest for grid / SCF residual noise. Ground-state E_UKS may be
checked at ~1e-4 Ha as a sanity band; the primary observables are the
**excitation energies [eV]**.

## What the regression should check

- Public API: concrete `UKS` with `XCType::LSDA`, Frame I/O only — no `*Config`
  bag, no live PySCF.
- `UKS` converges (ROKS warm-start optional, same multi-minima caution as
  `uks-tda-o2-lsda`); **`UKS::tddft`** (not `tda`) returns ≥5 roots with
  `tda == false` and `alda == true`.
- Each root [eV] within **1e-4 eV** of the hard-coded pins above.
- Must **not** hard-gate against the TDA root-3 pin (8.403 eV).

## Fair wall twin (offline; not ctest)

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

## Literature pointers (method, not numbers)

- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995) — Casida equations (A,B).
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs full.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.
