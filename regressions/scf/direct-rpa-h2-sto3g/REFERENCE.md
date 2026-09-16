# H₂ RHF/STO-3G direct RPA correlation energy — reference values

## Method

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
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.run(frame);
cpu::scf::DirectRPAResult rpa = rhf.direct_rpa();
// rpa.e_corr, rpa.e_tot [Hartree]; rpa.omega_ha
```

## Geometry

H₂, r(H–H) = **0.74 Å**, gas-phase.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| H    | 0.0   | 0.0   | 0.0   |
| H    | 0.0   | 0.0   | 0.74  |

## Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells)

## Convergence / solver

- Ground-state SCF: `conv_tol = 1e-12`
- Screening / response: full dRPA spectrum
  (`n_states = n_occ · n_vir = 1` on H₂/STO-3G)
- Correlation: plasmon formula (exact; no frequency quadrature, no DF)

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py direct-rpa \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

(`direct-rpa` rebuilds the Furche plasmon formula on **PySCF RHF orbitals +
exact MO ERIs** — the same non-DF target Atomiverse implements. Stock
`gw.rpa.RPA` (DF + `nw=40` Legendre) is also reported under `e_corr_df_rpa_ha`
for comparison.)

### Verbatim twin (plasmon on PySCF orbitals)

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

### Pinned output (2026-07-23)

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

### Regression gates

- Primary: \(E_{\mathrm{c}}\) within **1e-6 Ha** of −0.020633073749 Ha
- \(E_{\mathrm{tot}}\) within 1e-6 Ha
- \(\Omega_1\), \(\mathrm{Tr}\,A\) within 1e-6 Ha
- \(E_{\mathrm{RHF}}\) within 1e-6 Ha
- Plasmon identity \(|E_c - \tfrac12(\Omega_1-\mathrm{Tr}A)| < 10^{-12}\) Ha
- Sanity: \(E_c^{\mathrm{dRPA}} < E_c^{\mathrm{MP2}}\) (more correlation than MP2)

## Performance

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

## Citations

- Furche, J. Chem. Phys. **114**, 5982 (2001), DOI 10.1063/1.1353585
  (plasmon formula for RPA correlation).
- Ren et al., New J. Phys. **14**, 053020 (2012) (frequency-domain dRPA).
- Zhu & Chan, J. Chem. Theory Comput. **17**, 727 (2021); PySCF `gw.rpa`
  (DF + imaginary-axis quadrature implementation).
- G0W0 twin in-tree: `regressions/scf/g0w0-h2-sto3g` (same dRPA poles).
