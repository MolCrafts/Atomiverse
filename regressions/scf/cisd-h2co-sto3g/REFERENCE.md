# H2CO CIS(D)/STO-3G — S1 excitation energy pin

Closed-shell **CIS(D)** (doubles correction on RHF TDA/CIS) for the lowest
singlet of formaldehyde / STO-3G. Product API: `RHF::cisd(n_states)` after
`run(frame)`.

Literature / theory:

- M. Head-Gordon, R. J. Rico, M. Oumi, T. J. Lee, *A doubles correction to
  electronic excited states from configuration interaction in the space of
  single substitutions*, Chem. Phys. Lett. **219**, 21 (1994).
  DOI [10.1016/0009-2614(94)00070-0](https://doi.org/10.1016/0009-2614(94)00070-0)
  — spin-orbital eqs. (8), (13), (14):  
  \(\omega^{\mathrm{CIS(D)}}=\omega^{\mathrm{CIS}}+U+V\) with  
  \(V=-\tfrac14\sum_{ijab}(\tilde u_{ij}^{ab})^2/(\Delta_{ij}^{ab}-\omega)\) and  
  \(U=\sum_{ia}b_i^a v_i^a\).
- Parent CIS = RHF-TDA (Hirata & Head-Gordon 1999).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
PySCF supplies RHF + TDA amplitudes/integrals; the Head-Gordon CIS(D)
correction is evaluated offline with the same spin-orbital formulas as the
engine (PySCF has no built-in CIS(D) product path).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂CO (formaldehyde), charge 0, singlet |
| Geometry (Å) | C(0,0,0); O(0,0,1.21); H(0,±0.943,−0.587) — `pin_scf` `h2co` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CIS | `pyscf.tdscf.TDA(mf)`, `nstates≥3`, `conv_tol=1e-12` |
| CIS(D) | Head-Gordon 1994 spin-orbital \(U+V\) on each TDA root |
| AO / MO | 12 AO; 8 occupied, 4 virtual spatial MOs |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py cisd \
    --geom h2co --basis sto-3g --nstates 3 --conv-tol 1e-12
```

Equivalent explicit script (Head-Gordon eqs. on PySCF TDA + MO ERI):

```python
import numpy as np
from pyscf import gto, scf, tdscf, ao2mo
HA_TO_EV = 27.211386245988
mol = gto.M(atom="""
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.210000
H  0.000000  0.943000 -0.587000
H  0.000000 -0.943000 -0.587000
""", basis="sto-3g", unit="Angstrom", verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
td = tdscf.TDA(mf); td.nstates = 3; td.conv_tol = 1e-12; td.kernel()
# ... build spin-orbital ⟨pq||rs⟩ from ao2mo.full, MP2 a_ij^ab,
# ũ_ij^ab (eq.8), v_i^a (eq.13), ω_CIS(D)=ω+U+V (eq.14);
# PySCF TDA X has Σ X² = 1/2 ⇒ spin-orb b_{iσ}^{aσ} = X_ia
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −112.353990369522 Ha | 1e-6 Ha (reference) |
| CIS S1 \(\omega\) | 0.157224592070 Ha = **4.278299102174 eV** | 1e-4 eV |
| CIS(D) S1 \(\omega\) | 0.167010118990 Ha = **4.544576854822 eV** | **1e-4 eV** (product gate) |
| Doubles corr. \(U+V\) (S1) | +0.009785526920 Ha ≈ +0.2663 eV | (diagnostic) |

CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)
(`units::kHartreeInEv`) — both sides of the eV gate use the same factor.

On this minimal basis the S1 doubles correction is a small **blue** shift
(\(+0.27\,\mathrm{eV}\)); with larger bases Head-Gordon et al. typically find a
red shift for the valence \(n\to\pi^*\) of CH₂O. The pin is the Head-Gordon
formula evaluated on the PySCF RHF/TDA reference above, not an experimental
band origin.

## Gates (product regression)

- CIS(D) S1: \(|\omega_{\mathrm{ours}}-\mathrm{pin}| < 10^{-4}\,\mathrm{eV}\).
- CIS S1 (parent): same 1e-4 eV gate (sanity that TDA matches the pin root).
- \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha.
