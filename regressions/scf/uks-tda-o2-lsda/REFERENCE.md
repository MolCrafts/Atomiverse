# O₂ UKS-LSDA/STO-3G spin-conserving TDA — S/T-manifold roots [eV]

## Method

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
- TDA: default PySCF TDA; `nstates = 5`

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py uks-tda \
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
td = tdscf.TDA(mf)
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

## Tolerance and rationale

**|ω_i − pin_i| < 1e-4 eV** (absolute) for the five lowest roots.

Rationale: open-shell UKS-TDA is a new product path (unrestricted Casida **A**
+ spin-resolved ALDA); 1e-4 eV is the job gate for this textbook STO-3G case
and is tight enough to catch kernel-sign / channel-block errors while remaining
honest for grid / SCF residual noise. Ground-state E_UKS may be checked at
~1e-4 Ha as a sanity band; the primary observables are the **excitation
energies [eV]**.

## Literature pointers (method, not numbers)

- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response.
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF.
