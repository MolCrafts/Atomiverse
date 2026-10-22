# scf/eom-cc

One case; every section below is run by `atv_regressions --case scf/eom-cc` and keeps its own oracle record.


## H2 EOM-EE-CCSD/STO-3G — singlet S1 excitation energy pin

Closed-shell **EOM-EE-CCSD** on H₂ / STO-3G. Product API:
`CCSD<RHF>::eom_ee_ccsd(frame, n_states)` (drives RHF + spin-orbital CCSD + EOM-EE;
singlet roots selected from the spin-orbital Jacobian).

Literature / theory:

- J. F. Stanton & R. J. Bartlett, *The equation of motion coupled-cluster
  method. A systematic biorthogonal approach to molecular excitation energies,
  transition probabilities, and excited state properties*, J. Chem. Phys.
  **98**, 7029 (1993). DOI [10.1063/1.464746](https://doi.org/10.1063/1.464746)
- Z. Wang, Z. Tu & F. Wang, *Equation-of-Motion Coupled-Cluster Theory for
  Electron Attachment and Ionization …*, J. Chem. Theory Comput. **10**, 5567
  (2014) — EE σ equations (9)–(10) as coded in PySCF `eom_gccsd`.
- For two-electron systems EOM-EE-CCSD coincides with FCI excitation energies
  in the same basis.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12` |
| EOM | `mycc.eomee_ccsd_singlet(nroots=1)` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-ee-ccsd \
    --geom h2 --basis sto-3g --nstates 1 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, cc
HA_TO_EV = 27.211386245988
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
mycc = cc.CCSD(mf); mycc.conv_tol = 1e-12; mycc.kernel()
e, v = mycc.eomee_ccsd_singlet(nroots=1)
print(float(e[0]), float(e[0]) * HA_TO_EV, mycc.e_tot)
```

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{CCSD}}\) | −1.137283834497130 Ha | 1e-6 Ha |
| \(\omega_{\mathrm{S1}}\) (singlet EOM-EE) | **26.365966612792256 eV** | **1e-4 eV** (product gate) |
| \(\omega_{\mathrm{S1}}\) | 0.968931401526066 Ha | (cross-check) |

FCI excitation ladder on this basis (eV): 16.504 (triplet), **26.366 (singlet)**,
44.094 — EOM-EE-CCSD matches FCI for 2e.

### Gates (product regression)

- \(|\omega_{\mathrm{S1,ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\)
- \(|E_{\mathrm{CCSD,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

### Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-ee-ccsd --geom h2 --basis sto-3g --nstates 1 --repeat 3 -- \
    ./build/regressions/reg_eom-ee-ccsd
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.0725 s |
| Atomiverse median wall | 0.0140 s |
| **ratio_atv_over_pyscf** | **0.193** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.


## H2O EOM-IP-CCSD/STO-3G — lowest ionization potential pin

Closed-shell **EOM-IP-CCSD** on H₂O / STO-3G. Product API:
`CCSD<RHF>::eom_ip_ccsd(frame, n_states)` (drives RHF + spin-orbital CCSD + EOM-IP;
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

### Geometry and settings

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

### Repro command

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

### Pinned values

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

### Gates (product regression)

- \(|\mathrm{IP}_{0,\mathrm{ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\)
- \(|E_{\mathrm{CCSD,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

### Performance

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


## H2 EOM-EA-CCSD/STO-3G — lowest electron-attachment energy pin

Closed-shell **EOM-EA-CCSD** on H₂ / STO-3G. Product API:
`CCSD<RHF>::eom_ea_ccsd(frame, n_states)` (drives RHF + spin-orbital CCSD + EOM-EA;
unique attachment roots from the spin-orbital Jacobian, spin degeneracy
de-duplicated).

Literature / theory:

- M. Nooijen & R. J. Bartlett, *Equation of motion coupled cluster method for
  electron attachment*, J. Chem. Phys. **102**, 3629 (1995).
  DOI [10.1063/1.468591](https://doi.org/10.1063/1.468591)
- J. F. Stanton & R. J. Bartlett, *The equation of motion coupled-cluster
  method…*, J. Chem. Phys. **98**, 7029 (1993).
  DOI [10.1063/1.464746](https://doi.org/10.1063/1.464746)
- Spin-orbital σ equations as coded in PySCF `eom_gccsd.eaccsd_matvec`
  (Nooijen–Bartlett Eqs. (30)–(31)).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12` |
| EOM | `mycc.eaccsd(nroots=1)` (spatial RCCSD EA; matches GCCSD EOMEA) |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-ea-ccsd \
    --geom h2 --basis sto-3g --nstates 1 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, cc
HA_TO_EV = 27.211386245988
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
mycc = cc.CCSD(mf); mycc.conv_tol = 1e-12; mycc.kernel()
e, v = mycc.eaccsd(nroots=1)
print(float(e[0]), float(e[0]) * HA_TO_EV, mycc.e_tot)
```

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{CCSD}}\) | −1.137283834497130 Ha | 1e-6 Ha |
| \(\mathrm{EA}_0\) (lowest EOM-EA) | **18.821245619413340 eV** | **1e-4 eV** (product gate) |
| \(\mathrm{EA}_0\) | 0.691668019014956 Ha | (cross-check) |

H₂/STO-3G attachment is unbound (positive EA energy relative to neutral CCSD);
the LUMO Koopmans estimate is ~18.26 eV.

### Gates (product regression)

- \(|\mathrm{EA}_{0,\mathrm{ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\)
- \(|E_{\mathrm{CCSD,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

### Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-ea-ccsd --geom h2 --basis sto-3g --nstates 1 --repeat 3 -- \
    ./build/regressions/reg_eom-ea-ccsd
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.9293 s |
| Atomiverse median wall | 0.0635 s |
| **ratio_atv_over_pyscf** | **0.068** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.


## O₂ EOM-SF-CCSD/STO-3G — spin-flip excitation energies

Open-shell **EOM-SF-CCSD** on triplet O₂ / STO-3G. Product API:
`CCSD<UHF>::eom_sf_ccsd(frame, n_states, guess)` (drives high-spin UHF + UCCSD +
spin-flip EOM; ROHF density warm-start for the multi-minima UHF surface).

Literature / theory:

- A. I. Krylov, *Size-consistent wave functions for bond-breaking: the
  equation-of-motion spin-flip model*, Chem. Phys. Lett. **338**, 375 (2001).
  DOI [10.1016/S0009-2614(01)00287-1](https://doi.org/10.1016/S0009-2614(01)00287-1)
- S. V. Levchenko & A. I. Krylov, *Equation-of-motion spin-flip coupled-cluster
  model with single and double substitutions: Theory and application to
  cyclobutadiene*, J. Chem. Phys. **120**, 175 (2004).
  DOI [10.1063/1.1630018](https://doi.org/10.1063/1.1630018)
- PySCF implementation: `pyscf.cc.eom_uccsd.EOMEESpinFlip` /
  `UCCSD.eomsf_ccsd`.

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

### Geometry and settings

| Field | Value |
|---|---|
| Molecule | O₂, charge 0, triplet (`spin=2` = 2S) |
| Geometry (Å) | O(0,0,0); O(0,0,1.2075) — `pin_scf` `o2` preset |
| Basis | STO-3G |
| Angular convention | `cart=True` |
| SCF | UHF, `conv_tol=1e-12`, ROHF warm-start in ATV |
| CCSD | `pyscf.cc.UCCSD(mf)`, `conv_tol=1e-10` |
| EOM | `mycc.eomsf_ccsd(nroots=15)` then take the **lowest 5** real roots |
| Electrons | n_α=9, n_β=7 |
| AO / MO | 10 spatial AO/MO |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

**Davidson note:** `eomsf_ccsd(nroots=5)` alone can miss intermediate SF roots
(jumps from 1.59 eV to 8.81 eV). Requesting ≥10–15 roots recovers the full
lowest ladder; ATV uses a dense non-symmetric eigen-solve of the spin-orbital
Jacobian with majority-channel (α→β) SF selection, which matches that full
lowest ladder.

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-sf-ccsd \
    --geom o2 --basis sto-3g --spin 2 --nstates 15 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, cc
import numpy as np
HA_TO_EV = 27.211386245988
mol = gto.M(atom="""
O  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.207500
""", basis="sto-3g", unit="Angstrom", cart=True, spin=2, verbose=0)
mf = scf.UHF(mol); mf.conv_tol = 1e-12; mf.kernel()
mycc = cc.UCCSD(mf); mycc.conv_tol = 1e-10; mycc.kernel()
e, v = mycc.eomsf_ccsd(nroots=15)
e = np.sort(np.atleast_1d(np.asarray(e, dtype=float)))
print(mycc.e_tot, (e[:5] * HA_TO_EV).tolist())
```

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{UHF}}\) | −147.633946785450 Ha | 1e-6 Ha |
| \(E_{\mathrm{UCCSD}}\) | −147.741936323434 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) | −0.107989537984 Ha | 1e-6 Ha |
| \(\omega_0\) (SF, near-zero multiplet) | **0.004154770937 eV** | **1e-4 eV** |
| \(\omega_1\) | **1.024028019354 eV** | **1e-4 eV** |
| \(\omega_2\) | **1.024028354230 eV** | **1e-4 eV** |
| \(\omega_3\) | **1.586396740693 eV** | **1e-4 eV** |
| \(\omega_4\) | **6.184403293079 eV** | **1e-4 eV** |

Product gate focuses on \(\omega_0\ldots\omega_4\) at 1e-4 eV and UCCSD energy
at 1e-6 Ha.

### Gates (product regression)

- \(|\omega_{k,\mathrm{ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\) for \(k=0..4\)
- \(|E_{\mathrm{UCCSD,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

### Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-sf-ccsd --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 3 -- \
    ./build/regressions/reg_eom-sf-ccsd-o2
```

| Field | Value |
|---|---|
| Protocol | median of ≥3 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.317 s |
| Atomiverse median wall | 0.988 s |
| **ratio_atv_over_pyscf** | **3.12** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

ATV path: UHF (ROHF warm-start) + UCCSD + majority-SF dense Jacobian
(factored matvec intermediates). PySCF twin: UHF + UCCSD + `eomsf_ccsd(nroots=5)`
(Davidson; misses some intermediate SF roots that dense recovers — pins use
`nroots=15` lowest 5 for the accuracy gate).


## H2 EOM-CC2/STO-3G — singlet S1 excitation energy pin

Closed-shell **EOM-CC2** on H₂ / STO-3G. Product API:
`CC2<RHF>::eom_cc2(frame, n_states)` (drives RHF + spin-orbital CC2 + EOM-EE
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

### Geometry and settings

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

### Repro command

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

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{CC2}}\) | −1.129897380985958 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CC2) | −0.013138073589533 Ha | 1e-6 Ha |
| \(\omega_{\mathrm{S1}}\) (singlet EOM-CC2) | **26.164970973311735 eV** | **1e-4 eV** (product gate) |
| \(\omega_{\mathrm{S1}}\) | 0.961544948014894 Ha | (cross-check) |

Cross-check: full EOM-EE-CCSD on the same geometry gives S1 = 26.3659666 eV
(distinct; CC2 amplitudes shift the root by ~0.20 eV).

### Gates (product regression)

- \(|\omega_{\mathrm{S1,ours}} - \mathrm{pin}| < 10^{-4}\,\mathrm{eV}\)
- \(|E_{\mathrm{CC2,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- `converged == true`

### Performance

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
