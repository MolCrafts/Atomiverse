# H2 EOM-CC2/STO-3G — singlet S1 excitation energy pin

Closed-shell **EOM-CC2** on H₂ / STO-3G. Product API:
`RHF::eom_cc2(frame, n_states)` (drives RHF + spin-orbital CC2 + EOM-EE
Jacobian at the CC2 amplitudes; singlet roots selected from the spin-orbital
Jacobian).

Literature / theory:

- O. Christiansen, H. Koch, P. Jørgensen, *The second-order approximate
  coupled cluster singles and doubles model CC2*, Chem. Phys. Lett. **243**,
  409 (1995). DOI [10.1016/0009-2614(95)00841-Q](https://doi.org/10.1016/0009-2614(95)00841-Q)
- J. F. Stanton & R. J. Bartlett, *The equation of motion coupled-cluster
  method…*, J. Chem. Phys. **98**, 7029 (1993).
- Z. Wang, Z. Tu & F. Wang, *Equation-of-Motion Coupled-Cluster Theory for
  Electron Attachment and Ionization …*, J. Chem. Theory Comput. **10**, 5567
  (2014) — EE σ equations as coded in PySCF `eom_gccsd`.
- Stock PySCF has no top-level `EOMCC2` class; the product oracle is
  `pyscf.cc.rccsd.RCCSD` with `cc2=True`, then `eomee_ccsd_singlet` (full
  EOM-EE Jacobian evaluated at the CC2 T₁/T₂).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CC2 | `pyscf.cc.rccsd.RCCSD(mf); my.cc2=True`, `conv_tol=1e-12` |
| EOM | `mycc.eomee_ccsd_singlet(nroots=1)` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-cc2 \
    --geom h2 --basis sto-3g --nstates 1 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf
from pyscf.cc import rccsd
import numpy as np
HA_TO_EV = 27.211386245988
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
mycc = rccsd.RCCSD(mf); mycc.conv_tol = 1e-12; mycc.conv_tol_normt = 1e-12
mycc.cc2 = True
mycc.kernel()
e, v = mycc.eomee_ccsd_singlet(nroots=1)
w = float(np.asarray(e).ravel()[0])
print(w, w * HA_TO_EV, mycc.e_tot)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{CC2}}\) | −1.129897380985958 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CC2) | −0.013138073589533 Ha | 1e-6 Ha |
| \(\omega_{\mathrm{S1}}\) (singlet EOM-CC2) | **26.164970973311735 eV** | **1e-4 eV** (product gate) |
| \(\omega_{\mathrm{S1}}\) | 0.961544948014894 Ha | (cross-check) |

Cross-check: full EOM-EE-CCSD on the same geometry gives S1 = 26.3659666 eV
(distinct; CC2 amplitudes shift the root by ~0.20 eV).

## Gates (product regression)

- \(|\omega_{\mathrm{S1,ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\)
- \(|E_{\mathrm{CC2,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-cc2 --geom h2 --basis sto-3g --nstates 1 --repeat 3 -- \
    ./build/regressions/reg_eom-cc2-h2
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.0779 s |
| Atomiverse median wall | 0.0234 s |
| **ratio_atv_over_pyscf** | **0.300** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.
