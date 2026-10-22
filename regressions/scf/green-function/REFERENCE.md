# scf/green-function

One case; every section below is run by `atv_regressions --case scf/green-function` and keeps its own oracle record.


## H₂ RHF/STO-3G spectral G0W0 quasiparticle gaps — reference values

### Method

Closed-shell **G0W0** on an **RHF** reference: frequency-exact (spectral)
correlation self-energy with **direct RPA (dRPA)** screening of \(W\).
Observable: HOMO / LUMO quasiparticle energies and the **HOMO–LUMO QP gap
[eV]** (also IP \(= -\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\),
EA \(= \varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\)).

This is the Green's-function foothold for roadmap **G0W0** (excited-states /
many-body table). It is **not** TDDFT, **not** TDA/CIS, and **not** the
analytic-continuation (AC) `gw.GW` / `GWAC` path (AC numbers are recorded
only as a near twin; the regression pins the spectral / exact-frequency
solver).

Public API:

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-12};
rhf.run(frame);
cpu::scf::G0W0Result gw = cpu::scf::G0W0<cpu::scf::RHF>{rhf}.run();
// gw.gap_ev, gw.ip_ev, gw.ea_ev, gw.qp_energy_ha
```

### Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells)

### Convergence / solver

- Ground-state SCF: `conv_tol = 1e-12`
- Screening: full dRPA spectrum (`n_states = n_occ · n_vir = 1` on H₂/STO-3G)
- Self-energy: spectral sum, broadening \(\eta = 10^{-8}\) Ha
- QP equation: non-linear Newton, tolerance \(10^{-8}\) Ha (not linearized)

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py g0w0 \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

(`g0w0` builds `dft.RKS(mol, xc='hf')` then `gw.gw_exact.GWExact` + `tdscf.dRPA`
— spectral twin of Atomiverse `G0W0<RHF>::run()`. Default `gw.GW` / AC is also
reported under `gap_ac_ev` for comparison.)

#### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, dft, tdscf
from pyscf.gw import gw_exact
import pyscf
HA2EV = 27.211386245988
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
            unit="Angstrom", cart=True, verbose=0)
mf = dft.RKS(mol); mf.xc = "hf"; mf.conv_tol = 1e-12
e = mf.kernel()
td = tdscf.dRPA(mf); td.nstates = 1; td.kernel()
ge = gw_exact.GWExact(mf, tdmf=td); ge.verbose = 0; ge.kernel()
print("pyscf", pyscf.__version__)
print("E_RHF", e)
print("QP", ge.mo_energy)
print("gap_eV", (ge.mo_energy[1]-ge.mo_energy[0])*HA2EV)
print("IP_eV", -ge.mo_energy[0]*HA2EV)
print("EA_eV", ge.mo_energy[1]*HA2EV)
EOF
```

#### Pinned output (2026-07-23)

| Quantity | Value |
|----------|-------|
| PySCF | 2.14.0 |
| \(E_{\mathrm{RHF}}\) | **−1.116759307396** Ha |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{HF}}\) | −0.578553859832 Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{HF}}\) | +0.671143491914 Ha |
| HF gap | 34.005997328943 eV |
| dRPA \(\Omega_1\) | 1.570852128278 Ha = 42.745063997913 eV |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{G0W0}}\) | **−0.596957673490** Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{G0W0}}\) | **+0.689547305573** Ha |
| **QP gap** | **35.007583892680 eV** |
| IP \(-\varepsilon_H^{\mathrm{QP}}\) | **16.244045825851 eV** |
| EA \(\varepsilon_L^{\mathrm{QP}}\) | **18.763538066829 eV** |
| AC twin gap (`gw.GW`) | 35.007177297967 eV (info only; ~0.4 meV off spectral) |

#### Regression gates

- Primary: QP gap within **1e-4 eV** of 35.007583892680 eV
- IP / EA within 1e-4 eV
- QP orbital energies within 1e-5 Ha
- \(E_{\mathrm{RHF}}\) within 1e-6 Ha
- Sanity: G0W0 gap > HF gap on this system

### Performance

Fair bench vs PySCF spectral twin (same geom/basis/method class):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method g0w0 --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_g0w0-h2-sto3g
'
```

Recorded 2026-07-23 (median of 5; `OMP_NUM_THREADS=1`):

| Field | Value |
|-------|-------|
| \(T_{\mathrm{pyscf}}\) median | 0.1148 s |
| \(T_{\mathrm{atv}}\) median | 0.0125 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.109** |
| within_10x | **true** (≪ 10.0) |
| max_ratio | 10.0 |
| notes | Regression binary includes RHF + spectral G0W0; PySCF twin is RKS(hf)+dRPA+GWExact |

### Citations

- Hybertsen & Louie, Phys. Rev. B **34**, 5390 (1986), DOI 10.1103/PhysRevB.34.5390.
- van Setten et al., J. Chem. Theory Comput. **11**, 5665 (2015).
- PySCF GW: Zhu & Chan, J. Chem. Theory Comput. **17**, 727 (2021) (AC path);
  Berkelbach et al. spectral / exact-frequency implementation in `pyscf.gw.gw_exact`.


## H₂ RHF/STO-3G singlet BSE@G0W0 S1 — reference values

### Method

Closed-shell **Bethe–Salpeter equation (BSE)** on **spectral G0W0**
quasiparticle poles of an **RHF** reference. Static **direct-RPA** screened
Coulomb \(W\) (exact 4-centre MO ERIs; no density fitting). Observable:
lowest singlet excitation **S1 [eV]**.

This is the Green's-function excited-state foothold after
`reg_g0w0-h2-sto3g`. It is **not** TDDFT/TDHF and **not** the DF-only stock
`pyscf.gw.bse.BSE(gw.GW(...))` path (DF numbers are recorded as a near twin;
the regression pins the exact-ERI + spectral-G0W0 twin of `BSE<RHF>::run()`).

Public API:

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.run(frame);
cpu::scf::BSEResult bse = cpu::scf::BSE<cpu::scf::RHF>{rhf}.run(/*n_states=*/1); // full BSE
// bse.energy_ev[0]  // S1 [eV]
// bse.gw.gap_ev     // underlying G0W0 QP gap [eV]
```

Working equations (singlet, closed-shell spatial MOs; PySCF `gw.bse` /
Hillenbrand–Li–Zhu 2025):

\[
\begin{aligned}
A_{ia,jb} &= (\varepsilon_a^{\mathrm{QP}}-\varepsilon_i^{\mathrm{QP}})
            \delta_{ij}\delta_{ab} + 2(ia|jb) - W_{ij,ab}, \\
B_{ia,jb} &= 2(ia|jb) - W_{ib,aj}, \\
W &= v + v\,\Pi\,v,\quad
\Pi = 4\,(D^{-1}-4v_{\mathrm{ph}})^{-1},\quad
D^{-1}_{ia}=\varepsilon_i^{\mathrm{QP}}-\varepsilon_a^{\mathrm{QP}}.
\end{aligned}
\]

Full BSE solves the Casida RPA problem; BSE-TDA diagonalizes \(A\) only.

### Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells)

### Convergence / solver

- Ground-state SCF: `conv_tol = 1e-12`
- G0W0: spectral / frequency-exact (`gw.gw_exact.GWExact` + dRPA), same as
  `reg_g0w0-h2-sto3g`
- BSE: dense full diagonalization (`n_ov = n_occ·n_vir = 1` on H₂/STO-3G)
- Screening: static dRPA \(W\) with **QP** orbital energies (PySCF BSE
  convention)

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py bse \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

(`bse` builds spectral G0W0 poles then the exact-ERI static-RPA BSE twin of
`BSE<RHF>::run()`. Stock DF `pyscf.gw.bse.BSE` is also reported under
`s1_df_ev` for comparison only.)

#### Verbatim twin (exact-ERI BSE@spectral-G0W0)

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, dft, tdscf, ao2mo
from pyscf.gw import gw_exact
import numpy as np, pyscf
HA2EV = 27.211386245988
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
            unit="Angstrom", cart=True, verbose=0)
mf = dft.RKS(mol); mf.xc = "hf"; mf.conv_tol = 1e-12
e = mf.kernel()
C = mf.mo_coeff; nmo = C.shape[1]; nocc = mol.nelectron // 2; nvir = nmo - nocc
eri = ao2mo.restore(1, ao2mo.full(mol, C), nmo)
td = tdscf.dRPA(mf); td.nstates = nocc * nvir; td.kernel()
ge = gw_exact.GWExact(mf, tdmf=td); ge.verbose = 0; ge.kernel()
eps = np.asarray(ge.mo_energy)
nov = nocc * nvir
## OV index p = a*nocc + i (Atomiverse)
S = np.zeros((nov, nov)); vph = np.zeros((nov, nov))
for a in range(nvir):
    for i in range(nocc):
        p = a * nocc + i
        S[p, p] = eps[i] - eps[nocc + a]
        for b in range(nvir):
            for j in range(nocc):
                q = b * nocc + j
                v = eri[i, nocc + a, j, nocc + b]
                vph[p, q] = v
                S[p, q] -= 4.0 * v
Pi = 4.0 * np.linalg.inv(S)
def W(p, q, r, s):
    acc = eri[p, q, r, s]
    for a in range(nvir):
        for i in range(nocc):
            ia = a * nocc + i
            for b in range(nvir):
                for j in range(nocc):
                    jb = b * nocc + j
                    acc += (eri[p, q, i, nocc + a] * Pi[ia, jb]
                            * eri[j, nocc + b, r, s])
    return acc
A = np.zeros((nov, nov)); B = np.zeros((nov, nov))
for a in range(nvir):
    for i in range(nocc):
        ia = a * nocc + i; aa = nocc + a
        for b in range(nvir):
            for j in range(nocc):
                jb = b * nocc + j; bb = nocc + b
                vc = eri[i, aa, j, bb]
                A[ia, jb] = 2 * vc - W(i, j, aa, bb)
                B[ia, jb] = 2 * vc - W(i, bb, aa, j)
                if ia == jb:
                    A[ia, jb] += eps[aa] - eps[i]
from scipy.linalg import sqrtm
ApB = 0.5 * ((A + B) + (A + B).T)
AmB = 0.5 * ((A - B) + (A - B).T)
M = sqrtm(AmB) @ ApB @ sqrtm(AmB)
w = np.sqrt(np.linalg.eigvalsh(M.real))
print("pyscf", pyscf.__version__)
print("E_RHF", e)
print("QP", eps)
print("S1_eV", w[0] * HA2EV)
print("TDA_eV", np.linalg.eigvalsh(A)[0] * HA2EV)
EOF
```

#### Pinned output (2026-07-23)

| Quantity | Value |
|----------|-------|
| PySCF | 2.14.0 |
| \(E_{\mathrm{RHF}}\) | **−1.116759307396** Ha |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{G0W0}}\) | −0.596957673490 Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{G0W0}}\) | +0.689547305573 Ha |
| G0W0 QP gap | 35.007583892680 eV |
| **BSE S1** (full, exact ERI) | **25.956271542931 eV** |
| BSE-TDA S1 | 26.809052342076 eV |
| DF stock twin `gw.bse.BSE` S1 | ~25.954 eV (info only; ~2 meV DF error) |

#### Regression gates

- Primary: BSE S1 within **1e-4 eV** of 25.956271542931 eV
- BSE-TDA S1 within 1e-4 eV
- Underlying G0W0 gap within 1e-4 eV of 35.007583892680 eV
- \(E_{\mathrm{RHF}}\) within 1e-6 Ha
- Sanity: full BSE < TDA; BSE S1 < QP gap

### Performance

Fair bench vs PySCF exact-ERI twin (same geom/basis/method class):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method bse --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_bse-h2-sto3g
'
```

Recorded 2026-07-23 (median of 5; `OMP_NUM_THREADS=1`):

| Field | Value |
|-------|-------|
| \(T_{\mathrm{pyscf}}\) median | 0.2233 s |
| \(T_{\mathrm{atv}}\) median | 0.0134 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.060** |
| within_10x | **true** (≪ 10.0) |
| max_ratio | 10.0 |
| notes | Regression binary includes RHF + spectral G0W0 + BSE; PySCF twin is RKS(hf)+dRPA+GWExact+exact-ERI static BSE |

### Citations

- Hillenbrand, Li & Zhu, J. Chem. Phys. **162**, 174117 (2025) (PySCF BSE).
- J. Comput. Chem. **38**, 383 (2017) (molecular BSE / static W).
- Hybertsen & Louie, Phys. Rev. B **34**, 5390 (1986) (GW quasiparticles).
- PySCF BSE module: `pyscf.gw.bse` (DF production path; exact-ERI twin above).


## H₂ RHF/STO-3G direct RPA correlation energy — reference values

### Method

Closed-shell **direct RPA (dRPA)** correlation energy on an **RHF** reference:
Coulomb-only Casida \(A,B\) (\(\alpha_{\mathrm{hf}} = 0\)), correlation via the
**Furche plasmon formula**
\[
  E_{\mathrm{c}}^{\mathrm{dRPA}}
    = \tfrac12\Bigl(\sum_s \Omega_s - \mathrm{Tr}\,A\Bigr)
\]
with \(\Omega_s\) the positive eigenvalues of the Hermitian reduction
\((A-B)^{1/2}(A+B)(A-B)^{1/2}\). Equivalent to the exact imaginary-frequency
integral \(\frac{1}{2\pi}\int_0^\infty\mathrm{d}\omega\,
\mathrm{Tr}[\ln(1-\chi_0 v)+\chi_0 v]\) **without density fitting**.

This is the G0W0 screening twin as a **standalone correlation energy**
(roadmap Direct RPA). It is **not** TDHF excitation RPA (`RHF::tdhf`), **not**
TDDFT Casida, and **not** the DF+Legendre path of stock PySCF `gw.rpa.RPA`
(that path is recorded as a near twin; the regression pins the exact MO-ERI
plasmon value).

Public API:

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.run(frame);
cpu::scf::DirectRPAResult rpa = cpu::scf::DirectRPA<cpu::scf::RHF>{rhf}.run();
// rpa.e_corr, rpa.e_tot [Hartree]; rpa.omega_ha
```

### Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells)

### Convergence / solver

- Ground-state SCF: `conv_tol = 1e-12`
- Screening / response: full dRPA spectrum
  (`n_states = n_occ · n_vir = 1` on H₂/STO-3G)
- Correlation: plasmon formula (exact; no frequency quadrature, no DF)

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py direct-rpa \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

(`direct-rpa` rebuilds the Furche plasmon formula on **PySCF RHF orbitals +
exact MO ERIs** — the same non-DF target Atomiverse implements. Stock
`gw.rpa.RPA` (DF + `nw=40` Legendre) is also reported under `e_corr_df_rpa_ha`
for comparison.)

#### Verbatim twin (plasmon on PySCF orbitals)

```bash
~/work/.x86_64/bin/python - <<'EOF'
import numpy as np
from numpy.linalg import eigh
from scipy.linalg import sqrtm
from pyscf import gto, scf, ao2mo
import pyscf
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
            unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12
e_hf = mf.kernel()
C, eps = mf.mo_coeff, mf.mo_energy
nmo, nocc = C.shape[1], mol.nelectron // 2
nvir = nmo - nocc
eri = ao2mo.restore(1, ao2mo.full(mol, C), nmo)
n_ov = nocc * nvir
A = np.zeros((n_ov, n_ov)); B = np.zeros_like(A)
for i in range(nocc):
    for a in range(nvir):
        ia = i * nvir + a; aa = nocc + a
        A[ia, ia] = eps[aa] - eps[i]
        for j in range(nocc):
            for b in range(nvir):
                jb = j * nvir + b; bb = nocc + b
                v = 2 * eri[i, aa, j, bb]
                A[ia, jb] += v; B[ia, jb] += v
Mm, Mp = A - B, A + B
T = np.real(sqrtm(Mm))
M = 0.5 * ((T @ Mp @ T) + (T @ Mp @ T).T)
w2, _ = eigh(M)
Om = np.sqrt(np.maximum(w2, 0.0))
e_corr = 0.5 * (Om.sum() - np.trace(A))
print("pyscf", pyscf.__version__)
print("E_RHF", e_hf)
print("Omega", Om)
print("TrA", np.trace(A))
print("e_corr", e_corr)
print("e_tot", e_hf + e_corr)
EOF
```

#### Pinned output (2026-07-23)

| Quantity | Value |
|----------|-------|
| PySCF | 2.14.0 |
| \(E_{\mathrm{RHF}}\) | **−1.116759307396** Ha |
| dRPA \(\Omega_1\) | **1.570852128278** Ha (= 42.745063997913 eV; same pole as G0W0 screening) |
| \(\mathrm{Tr}\,A\) | **1.612118275776** Ha |
| **\(E_{\mathrm{c}}^{\mathrm{dRPA}}\)** | **−0.020633073749** Ha |
| \(E_{\mathrm{tot}}^{\mathrm{dRPA}}\) | **−1.137392381145** Ha |
| DF twin (`gw.rpa.RPA`, `nw=100`) | −0.020624247051 Ha (info only; ~9e-6 Ha DF residual) |
| RMP2 (context) | −0.013138073590 Ha |

#### Regression gates

- Primary: \(E_{\mathrm{c}}\) within **1e-6 Ha** of −0.020633073749 Ha
- \(E_{\mathrm{tot}}\) within 1e-6 Ha
- \(\Omega_1\), \(\mathrm{Tr}\,A\) within 1e-6 Ha
- \(E_{\mathrm{RHF}}\) within 1e-6 Ha
- Plasmon identity \(|E_c - \tfrac12(\Omega_1-\mathrm{Tr}A)| < 10^{-12}\) Ha
- Sanity: \(E_c^{\mathrm{dRPA}} < E_c^{\mathrm{MP2}}\) (more correlation than MP2)

### Performance

Fair bench vs the offline plasmon twin (same geom/basis/method class):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method direct_rpa --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_direct-rpa-h2-sto3g
'
```

Recorded 2026-07-23 (median of 5; `OMP_NUM_THREADS=1`):

| Field | Value |
|-------|-------|
| \(T_{\mathrm{pyscf}}\) median | 0.0217 s |
| \(T_{\mathrm{atv}}\) median | 0.0165 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.760** |
| within_10x | **true** (≪ 10.0) |
| max_ratio | 10.0 |
| notes | Regression binary includes RHF + dRPA plasmon; PySCF twin is RHF + exact MO-ERI plasmon (not DF `gw.rpa.RPA`) |

### Citations

- Furche, J. Chem. Phys. **114**, 5982 (2001), DOI 10.1063/1.1353585
  (plasmon formula for RPA correlation).
- Ren et al., New J. Phys. **14**, 053020 (2012) (frequency-domain dRPA).
- Zhu & Chan, J. Chem. Theory Comput. **17**, 727 (2021); PySCF `gw.rpa`
  (DF + imaginary-axis quadrature implementation).
- G0W0 twin in-tree: `regressions/scf/green-function` (same dRPA poles).


## H₂ RHF/STO-3G one-shot GF2 (G0F2) quasiparticle gaps — reference values

### Method

Closed-shell **one-shot second-order Green's function** (G0F2 / GF2) on an
**RHF** reference: the 2h1p + 2p1h second-order self-energy is built in the
auxiliary (Lehmann) representation from the HF Green's function, then the
Dyson equation is solved **once** with the HF Fock matrix
\(F = \mathrm{diag}(\varepsilon_{\mathrm{HF}})\). Observable: HOMO / LUMO
quasiparticle energies and the **HOMO–LUMO QP gap [eV]** (also
IP \(= -\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\),
EA \(= \varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\)), plus the Galitskii–Migdal
correlation energy (equals canonical MP2 on a pure HF reference).

This is the Green's-function foothold for roadmap **GF2** (excited-states /
many-body table). It is **not** G0W0 (screened \(W\); bare \(v\) here),
**not** ADC(2), and **not** self-consistent AGF2 (outer density/Fock loop —
separate roadmap row). The regression pins the **one-shot** path of PySCF
`agf2.RAGF2` (SE from `init_gf()` → `se.get_greens_function(F_HF)`).

Public API:

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.run(frame);
cpu::scf::GF2Result gf = cpu::scf::GF2<cpu::scf::RHF>{rhf}.run();
// gf.gap_ev, gf.ip_ev, gf.ea_ev, gf.qp_energy_ha, gf.e_corr
```

### Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

### Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells)

### Convergence / solver

- Ground-state SCF: `conv_tol = 1e-12`
- Self-energy: PySCF AGF2 construction (`os_factor = ss_factor = 1`,
  `weight_tol = 1e-11` to drop uncoupled poles)
- Dyson: single Hermitian diagonalization of the extended Fock matrix
  \([F, V; V^\dagger, \varepsilon_{\mathrm{aux}}]\)
- Chemical potential: binary search for \(N = 2\) electrons (RHF occupancy 2)

### Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

#### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py gf2 \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

#### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, scf, agf2
import pyscf, numpy as np
HA2EV = 27.211386245988
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
            unit="Angstrom", cart=True, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; e = mf.kernel()
g1 = agf2.RAGF2(mf); g1.verbose = 0
eri = g1.ao2mo()
g0 = g1.init_gf()
se0 = g1.build_se(eri, g0)
fock = np.diag(mf.mo_energy)
from pyscf.agf2.chempot import binsearch_chempot
w, v = se0.eig(fock)
mu, _ = binsearch_chempot((w, v), g1.nmo, g1.nocc * 2)
g_one = agf2.GreensFunction(w, v[:g1.nmo], chempot=mu)
ip = g1.get_ip(g_one, nroots=1)[0][0]
ea = g1.get_ea(g_one, nroots=1)[0][0]
e_init = g1.energy_mp2(mf.mo_energy, se0)
print("pyscf", pyscf.__version__)
print("E_RHF", e)
print("IP_Ha", ip, "EA_Ha", ea)
print("gap_eV", (ip + ea) * HA2EV)
print("e_init", e_init)
EOF
```

#### Pinned output (2026-07-23)

| Quantity | Value |
|----------|-------|
| PySCF | 2.14.0 |
| \(E_{\mathrm{RHF}}\) | **−1.116759307396** Ha |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{HF}}\) | −0.578553859832 Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{HF}}\) | +0.671143491914 Ha |
| HF gap | 34.005997328943 eV |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{GF2}}\) | **−0.591623589738** Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{GF2}}\) | **+0.684213221821** Ha |
| **QP gap** | **34.717288266166 eV** |
| IP \(-\varepsilon_H^{\mathrm{QP}}\) | **16.098898012595 eV** |
| EA \(\varepsilon_L^{\mathrm{QP}}\) | **18.618390253572 eV** |
| \(E_{\mathrm{corr}}\) (GM / MP2) | **−0.013138073590** Ha |

SC-AGF2 companion (info only; not the regression pin): full `RAGF2.kernel()`
gap ≈ 34.451051 eV, e_corr ≈ −0.012920 Ha.

#### Regression gates

- Primary: QP gap within **1e-4 eV** of 34.717288266166 eV
- IP / EA within 1e-4 eV
- QP orbital energies within 1e-5 Ha
- \(E_{\mathrm{RHF}}\) and \(E_{\mathrm{corr}}\) within 1e-6 Ha
- Sanity: gap > 0 and finite

### Performance

Fair bench vs PySCF one-shot twin (same geom/basis/method class):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method gf2 --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_gf2-h2-sto3g
'
```

Recorded 2026-07-23 (median of 5; `OMP_NUM_THREADS=1`):

| Field | Value |
|-------|-------|
| \(T_{\mathrm{pyscf}}\) median | 0.2449 s |
| \(T_{\mathrm{atv}}\) median | 0.0174 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.071** |
| within_10x | **true** (≪ 10.0) |
| max_ratio | 10.0 |
| notes | Regression binary includes RHF + one-shot GF2; PySCF twin is RHF+RAGF2 SE+Dyson |

### Citations

- Backhouse, Holzmann & Booth, J. Chem. Theory Comput. **17**, 3414 (2021)
  (PySCF auxiliary GF2).
- Galitskii & Migdal, Sov. Phys. JETP **7**, 96 (1958).
- Roadmap distinguishes **GF2** (one-shot / G0F2) from **self-consistent GF2**.
