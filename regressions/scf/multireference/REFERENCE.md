# scf/multireference

One case; every section below is run by `atv_regressions --case scf/multireference` and keeps its own oracle record.


## H2CO CIS(D)/STO-3G — S1 excitation energy pin

Closed-shell **CIS(D)** (doubles correction on RHF TDA/CIS) for the lowest
singlet of formaldehyde / STO-3G. Product API: `CISD<RHF>::run(n_states)` after
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

### Geometry and settings

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

### Repro command

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
## ... build spin-orbital ⟨pq||rs⟩ from ao2mo.full, MP2 a_ij^ab,
## ũ_ij^ab (eq.8), v_i^a (eq.13), ω_CIS(D)=ω+U+V (eq.14);
## PySCF TDA X has Σ X² = 1/2 ⇒ spin-orb b_{iσ}^{aσ} = X_ia
```

### Pinned values

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

### Gates (product regression)

- CIS(D) S1: \(|\omega_{\mathrm{ours}}-\mathrm{pin}| < 10^{-4}\,\mathrm{eV}\).
- CIS S1 (parent): same 1e-4 eV gate (sanity that TDA matches the pin root).
- \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha.


## H2 / STO-3G — unified determinant-CI engine equivalence

`RHF::fci` and `RHF::casci(2,2)` must both reproduce the offline PySCF full-CI
pin AND agree with each other to machine precision. On H2 / STO-3G the full CI
space and the CAS(2e,2o) space are identical, so FCI == CASCI(full space) is
exact. This is the numeric equivalence gate for scf-review-03-posthf-dedup: it
proves that unifying the determinant-CI machinery (formerly forked between
`fci.cpp` and `casscf.cpp`) into one engine changed no number and that both
public entry points route the same engine.

Product API (out-of-library consumer, public headers only):

```cpp
RHF::fci(frame);                       // full CI, drives a fresh RHF
RHF::casci(frame, /*ncas=*/2, /*nelecas=*/2);  // full valence CAS, drives a fresh RHF
```

Literature / theory:

- B. O. Roos, *The Complete Active Space Self-Consistent Field Method and its
  Applications in Electronic Structure Calculations*, Adv. Chem. Phys. **69**,
  399 (1987). DOI [10.1002/9780470142943.ch7](https://doi.org/10.1002/9780470142943.ch7)
- Szabo & Ostlund, *Modern Quantum Chemistry*, Dover (1996), §2.3 (Slater–Condon
  rules), ISBN 0-486-69186-1.
- On the two-electron / two-orbital H2 foothold, CASCI(2e,2o) is **identical** to
  full CI in the basis (and to CCSD / CISD).

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
| FCI | `pyscf.fci.FCI(mf)` (full space) |
| CASCI | `pyscf.mcscf.CASCI(mf, ncas=2, nelecas=2)` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO; ncore=0 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py casci \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

Equivalent (FCI == CASCI(full) on this system):

```python
from pyscf import gto, scf, mcscf, fci
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
e_fci = fci.FCI(mf).kernel()[0]
mc = mcscf.CASCI(mf, 2, 2); e_cas = mc.kernel()[0]
print(e_hf, e_fci, e_cas)   # e_fci == e_cas == -1.137283834488502
```

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{FCI}}\) (total) | −1.137283834488502 Ha | 1e-6 Ha |
| \(E_{\mathrm{CASCI}(2,2)}\) (total) | −1.137283834488502 Ha | 1e-6 Ha |
| \(|E_{\mathrm{FCI}} - E_{\mathrm{CASCI}}|\) | 0 | **1e-12 Ha** (equivalence gate) |

The full-CI total −1.137283834488502 Ha is the offline PySCF 2.14.0 literal
(`mc.e_tot` of `pyscf.mcscf.CASCI`); FCI coincides with CASCI(2,2) to ~1e-14 Ha
on this system.

### Gates (product regression)

- \(|E_{\mathrm{FCI}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{CASCI}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(|E_{\mathrm{FCI}} - E_{\mathrm{CASCI}}| < 10^{-12}\,\mathrm{Ha}\) (unified-engine equivalence)
- `ncore == 0`, `ncas == 2`, `nelecas == 2`
- runs in < 60 s; no live third-party call.


## H2 CASSCF(2,2)/STO-3G — total energy pin

Closed-shell **CASSCF** with a full valence active space on H₂ / STO-3G.
Product API: `RHF::casscf(frame, /*ncas=*/2, /*nelecas=*/2)` (drives a fresh
RHF, freezes no core, dense FCI in the 2-orbital active space; orbital
optimization has no non-redundant pairs when `ncas = n_mo`).

Literature / theory:

- B. O. Roos, *The Complete Active Space Self-Consistent Field Method and its
  Applications in Electronic Structure Calculations*, Adv. Chem. Phys. **69**,
  399 (1987). DOI [10.1002/9780470142943.ch7](https://doi.org/10.1002/9780470142943.ch7)
- On this two-electron / two-orbital foothold, CASSCF(2e,2o) is **identical** to
  CASCI / full CI in the basis (active–active rotations are redundant; no core
  or virtual outside the active window).

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
| CASSCF | `pyscf.mcscf.CASSCF(mf, ncas=2, nelecas=2)`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occupied, 1 virtual spatial MO; ncore=0 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py casscf \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, mcscf
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
mc = mcscf.CASSCF(mf, 2, 2); mc.conv_tol = 1e-12
e_mc = mc.kernel()[0]
print(e_hf, e_mc, e_mc - e_hf, mc.converged)
```

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (CASSCF) | −0.020524527092077 Ha | **1e-6 Ha** |
| \(E_{\mathrm{tot}}\) (CASSCF) | −1.137283834488502 Ha | **1e-6 Ha** (product gate) |

CASSCF(2,2) coincides with CASCI / FCI / CCSD to ~1e-14 Ha on this system
(full valence active space; zero non-redundant orbital rotations).

### Gates (product regression)

- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\) (primary gate)
- \(|E_{\mathrm{corr,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha
- `ncore == 0`, `ncas == 2`, `nelecas == 2`, `converged == true`
- CASSCF energy equals CASCI on the same reference to \(10^{-10}\) Ha

### Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method casscf --geom h2 --basis sto-3g --repeat 5 -- \
    ./build/regressions/reg_casscf-h2-sto3g
```

| Field | Value |
|---|---|
| Protocol | median of ≥5 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.2112 s |
| Atomiverse median wall | 0.0092 s |
| **ratio_atv_over_pyscf** | **0.0435** |
| **within_10x** | **true** |
| Bench date | 2026-07-23, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.


## H2 SC-NEVPT2 / CASSCF(2,2)/STO-3G — correlation pin

Closed-shell **strongly contracted NEVPT2** on a full-valence CASSCF(2,2)
reference for H₂ / STO-3G. Product API:
`RHF::nevpt2(frame, /*ncas=*/2, /*nelecas=*/2)` (drives RHF → CASSCF → SC-NEVPT2).

Literature / theory:

- C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu,
  *Introduction of n-electron valence states for multireference perturbation
  theory*, J. Chem. Phys. **114**, 10252 (2001).
  DOI [10.1063/1.1361246](https://doi.org/10.1063/1.1361246)
- C. Angeli, R. Cimiraglia, J.-P. Malrieu, *n-electron valence state
  perturbation theory: A spinless formulation and an efficient implementation
  of the strongly contracted and of the partially contracted variants*,
  J. Chem. Phys. **117**, 9138 (2002).
  DOI [10.1063/1.1515317](https://doi.org/10.1063/1.1515317)

On this two-electron / two-orbital foothold, CASSCF(2e,2o) spans **all** MOs
(`ncore = 0`, `nvirt = 0`). Every SC-NEVPT2 perturber class (Sr, Si, Sij, Srs,
Sir, Sijr, Srsi, Sijrs) requires at least one inactive or external index, so
the PT2 correction is **identically zero** and
\(E_{\mathrm{NEVPT2}} = E_{\mathrm{CASSCF}} = E_{\mathrm{FCI}}\).

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
| CASSCF | `pyscf.mcscf.CASSCF(mf, ncas=2, nelecas=2)`, `conv_tol=1e-12` |
| NEVPT2 | `pyscf.mrpt.NEVPT(mc).kernel()` (strongly contracted default) |
| AO / MO | 2 AO; ncore=0, ncas=2, nvirt=0 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-24, venv `~/work/.x86_64` |

### Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py nevpt2 \
    --geom h2 --basis sto-3g --ncas 2 --nelecas 2 --conv-tol 1e-12
```

Equivalent:

```python
from pyscf import gto, scf, mcscf, mrpt
mol = gto.M(atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""", basis="sto-3g", unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e_hf = mf.kernel()
mc = mcscf.CASSCF(mf, 2, 2); mc.conv_tol = 1e-12
e_mc = mc.kernel()[0]
e_pt2 = mrpt.NEVPT(mc).kernel()
print(e_hf, e_mc, e_pt2, e_mc + e_pt2, mc.ncore)
```

### Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | 1e-6 Ha |
| \(E_{\mathrm{CASSCF}}\) | −1.137283834488502 Ha | 1e-6 Ha |
| \(E_{\mathrm{corr}}\) (SC-NEVPT2 PT2) | **0.0** Ha | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}\) (NEVPT2) | −1.137283834488502 Ha | **1e-6 Ha** |

`e_corr` is the PT2 correction vs CASSCF (PySCF `mrpt.NEVPT.e_corr`), not
vs HF. On this system it is exactly zero by construction of the active space.

### Gates (product regression)

- \(|E_{\mathrm{corr,ours}} - 0| < 10^{-6}\,\mathrm{Ha}\) (primary gate)
- \(|E_{\mathrm{tot,ours}} - \mathrm{pin}| < 10^{-6}\,\mathrm{Ha}\)
- \(E_{\mathrm{CASSCF}}\), \(E_{\mathrm{RHF}}\): \(10^{-6}\) Ha
- `ncore == 0`, `nvirt == 0`, `ncas == 2`, `nelecas == 2`, `converged == true`
- NEVPT2 total equals CASSCF on the same reference to \(10^{-10}\) Ha

### Performance

Fair wall-time vs PySCF (offline; never ctest):

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method nevpt2 --geom h2 --basis sto-3g --repeat 5 -- \
    ./build/regressions/reg_nevpt2-h2-sto3g
```

| Field | Value |
|---|---|
| Protocol | median of ≥5 runs; ratio = T_atv / T_pyscf |
| Target | ratio ≤ 10.0 |
| PySCF median wall | 0.1725 s |
| Atomiverse median wall | 0.0125 s |
| **ratio_atv_over_pyscf** | **0.0723** |
| **within_10x** | **true** |
| Bench date | 2026-07-24, login node, OMP_NUM_THREADS=1 |

H₂/STO-3G is startup-dominated; ATV is faster than PySCF on this foothold
(ratio ≪ 1). No performance optimization pass required.
