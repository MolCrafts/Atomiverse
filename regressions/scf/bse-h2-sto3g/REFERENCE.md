# H₂ RHF/STO-3G singlet BSE@G0W0 S1 — reference values

## Method

Closed-shell **Bethe–Salpeter equation (BSE)** on **spectral G0W0**
quasiparticle poles of an **RHF** reference. Static **direct-RPA** screened
Coulomb \(W\) (exact 4-centre MO ERIs; no density fitting). Observable:
lowest singlet excitation **S1 [eV]**.

This is the Green's-function excited-state foothold after
`reg_g0w0-h2-sto3g`. It is **not** TDDFT/TDHF and **not** the DF-only stock
`pyscf.gw.bse.BSE(gw.GW(...))` path (DF numbers are recorded as a near twin;
the regression pins the exact-ERI + spectral-G0W0 twin of `RHF::bse()`).

Public API:

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.run(frame);
cpu::scf::BSEResult bse = rhf.bse(/*n_states=*/1); // full BSE
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
- G0W0: spectral / frequency-exact (`gw.gw_exact.GWExact` + dRPA), same as
  `reg_g0w0-h2-sto3g`
- BSE: dense full diagonalization (`n_ov = n_occ·n_vir = 1` on H₂/STO-3G)
- Screening: static dRPA \(W\) with **QP** orbital energies (PySCF BSE
  convention)

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py bse \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

(`bse` builds spectral G0W0 poles then the exact-ERI static-RPA BSE twin of
`RHF::bse()`. Stock DF `pyscf.gw.bse.BSE` is also reported under
`s1_df_ev` for comparison only.)

### Verbatim twin (exact-ERI BSE@spectral-G0W0)

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
# OV index p = a*nocc + i (Atomiverse)
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

### Pinned output (2026-07-23)

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

### Regression gates

- Primary: BSE S1 within **1e-4 eV** of 25.956271542931 eV
- BSE-TDA S1 within 1e-4 eV
- Underlying G0W0 gap within 1e-4 eV of 35.007583892680 eV
- \(E_{\mathrm{RHF}}\) within 1e-6 Ha
- Sanity: full BSE < TDA; BSE S1 < QP gap

## Performance

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

## Citations

- Hillenbrand, Li & Zhu, J. Chem. Phys. **162**, 174117 (2025) (PySCF BSE).
- J. Comput. Chem. **38**, 383 (2017) (molecular BSE / static W).
- Hybertsen & Louie, Phys. Rev. B **34**, 5390 (1986) (GW quasiparticles).
- PySCF BSE module: `pyscf.gw.bse` (DF production path; exact-ERI twin above).
