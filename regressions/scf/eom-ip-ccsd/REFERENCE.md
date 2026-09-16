# H2O EOM-IP-CCSD/STO-3G — lowest ionization potential pin

Closed-shell **EOM-IP-CCSD** on H₂O / STO-3G. Product API:
`RHF::eom_ip_ccsd(frame, n_states)` (drives RHF + spin-orbital CCSD + EOM-IP;
unique ionization roots from the spin-orbital Jacobian, spin degeneracy
de-duplicated).

Literature / theory:

- J. F. Stanton & R. J. Bartlett, *The equation of motion coupled-cluster
  method…*, J. Chem. Phys. **98**, 7029 (1993).
  DOI [10.1063/1.464746](https://doi.org/10.1063/1.464746)
- M. Nooijen & J. G. Snijders, *Coupled cluster approach to the single-particle
  Green's function*, Int. J. Quantum Chem. **S26**, 55 (1992).
- Z. Tu, F. Wang & X. Li, *Equation-of-motion coupled-cluster method for
  ionized states…*, J. Chem. Phys. **136**, 174102 (2012).
  DOI [10.1063/1.4704893](https://doi.org/10.1063/1.4704893)
- Spin-orbital σ equations as coded in PySCF `eom_gccsd.ipccsd_matvec`
  (Tu–Wang–Li Eqs. (8)–(9)).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂O (water), charge 0, singlet |
| Geometry (Å) | O(0,0,0.1173); H(0,±0.7572,−0.4692) — `pin_scf` `water` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12` |
| EOM | `mycc.ipccsd(nroots≥1)` (spatial RCCSD IP; matches GCCSD EOMIP after spin de-dupe) |
| AO / MO | 7 AO; 5 occupied, 2 virtual spatial MOs |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-ip-ccsd \
    --geom water --basis sto-3g --nstates 3 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, cc
import numpy as np
HA_TO_EV = 27.211386245988
mol = gto.M(atom="""
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
mycc = cc.CCSD(mf); mycc.conv_tol = 1e-12; mycc.kernel()
e, v = mycc.ipccsd(nroots=3)
e = np.atleast_1d(np.asarray(e, dtype=float))
print(float(e[0]), float(e[0]) * HA_TO_EV, mycc.e_tot)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{CCSD}}\) | −75.01246170149486 Ha | 1e-6 Ha |
| \(\mathrm{IP}_0\) (lowest EOM-IP) | **8.415727034061687 eV** | **1e-4 eV** (product gate) |
| \(\mathrm{IP}_0\) | 0.309272264117102 Ha | (cross-check) |
| \(\mathrm{IP}_1\) | ≈ 10.911018291684588 eV | (diagnostic) |
| \(\mathrm{IP}_2\) | ≈ 16.62369673430045 eV | (diagnostic) |

CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)
(`units::kHartreeInEv`) — both sides of the eV gate use the same factor.

On this minimal basis the lowest IP is ~8.42 eV (valence). The pin is the
PySCF EOM-IP-CCSD root above, not an experimental vertical IP.

## Gates (product regression)

- \(|\mathrm{IP}_{0,\mathrm{ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\)
- \(|E_{\mathrm{CCSD,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

## Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-ip-ccsd --geom water --basis sto-3g --nstates 1 --repeat 3 -- \
    ./build/regressions/reg_eom-ip-ccsd
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.1470 s |
| Atomiverse median wall | 0.1108 s |
| **ratio_atv_over_pyscf** | **0.754** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂O/STO-3G is still startup-dominated; ATV is slightly faster than PySCF on
this foothold (ratio < 1). No performance optimization pass required.
