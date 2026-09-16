# O₂ UHF/STO-3G collinear spin-flip full TDDFT (SF-RPA) — excitation roots [eV]

## Method

**Collinear spin-flip full linear response** (SF-TDHF / SF-RPA) on a high-spin
**UHF** reference (triplet O₂, \(2S=2\)). Ms-decreasing singles
\(i_\alpha \to a_\beta\) (majority→minority) form the response space — same
manifold as SF-TDA (`UHF::sf_tda`). With the collinear HF kernel the Coulomb
term vanishes for pure spin-flip and only exchange remains (Shao, Head-Gordon
& Krylov, *J. Chem. Phys.* **118**, 4807 (2003)):

\[
A_{ia,jb}
=
\delta_{ij}\delta_{ab}
(\varepsilon_a^\beta-\varepsilon_i^\alpha)
-(ij|ab)
\]

\[
B_{ia,jb}
=
-(ib|aj)
\]

Non-Hermitian RPA eigenproblem

\[
\begin{bmatrix}A&B\\-B&-A\end{bmatrix}
\begin{bmatrix}X\\Y\end{bmatrix}
=
\omega
\begin{bmatrix}X\\Y\end{bmatrix}
\]

solved densely (`dgeev`). On this O₂/STO-3G UHF reference \((A-B)\) is
**not** positive definite (lowest eigenvalue ≈ −0.46 Ha), so the Hermitian
Casida reduction used by closed-shell `RKS::tddft` is undefined; the product
path is the non-Hermitian form. Observable: lowest **positive real** SF-RPA
roots **ω [eV]**.

Public path: `UHF::sf_tddft(n_states)` after a converged `UHF::run`.

PySCF **mainline has no product SF-TDDFT**; the offline pin builds the same
\(A,B\) with PySCF UHF MOs + `ao2mo` (`tools/pyscf_oracle/pin_scf.py sf-tddft`).
That script is also the fair-bench twin (`--method sf_tddft`).

## Geometry

O₂, r(O–O) = **1.2075 Å**, charge 0, spin \(2S=2\) (triplet).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.0   | 0.0   | 0.0   |
| O    | 0.0   | 0.0   | 1.2075 |

## Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True`; STO-3G has no \(d\), cart flag inert)
- Occupation: \((n_\alpha, n_\beta) = (9, 7)\); SF space \(n_{ov}=9\times3=27\)

## Convergence

- Ground-state UHF: `conv_tol = 1e-12`
- SF-RPA: dense non-Hermitian `dgeev` of the \(2n_{ov}\) RPA matrix; `nstates = 5`

## Oracle: PySCF 2.14.0 integrals (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py sf-tddft \
  --geom o2 --basis sto-3g --spin 2 --nstates 5 --conv-tol 1e-12
```

### Verbatim twin

```bash
~/work/.x86_64/bin/python - <<'EOF'
from pyscf import gto, scf, ao2mo
import numpy as np
HA2EV = 27.211386245988
mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g", spin=2,
            unit="Angstrom", cart=True, verbose=0)
mf = scf.UHF(mol); mf.conv_tol = 1e-12; e = mf.kernel()
mo_a, mo_b = mf.mo_coeff; ea, eb = mf.mo_energy; occa, occb = mf.mo_occ
oa = np.where(occa>0)[0]; vb = np.where(occb==0)[0]
no, nv = len(oa), len(vb); nov = no*nv
Co, Cv = mo_a[:, oa], mo_b[:, vb]
ijab = ao2mo.general(mol, (Co, Co, Cv, Cv), compact=False).reshape(no, no, nv, nv)
ibaj = ao2mo.general(mol, (Co, Cv, Cv, Co), compact=False).reshape(no, nv, nv, no)
A = np.zeros((nov, nov)); B = np.zeros((nov, nov))
for i in range(no):
    for a in range(nv):
        p = i*nv + a
        A[p, p] = eb[vb[a]] - ea[oa[i]]
        for j in range(no):
            for b in range(nv):
                q = j*nv + b
                A[p, q] -= ijab[i, j, a, b]
                B[p, q] -= ibaj[i, b, a, j]
A = 0.5*(A+A.T); B = 0.5*(B+B.T)
M = np.block([[A, B], [-B, -A]])
w = np.linalg.eigvals(M)
pos = sorted([float(z.real) for z in w if z.real > 1e-8 and abs(z.imag) <= 1e-8])
print("E_UHF = %.12f Ha  S2=%.10f  A-B_min=%.6f" % (e, mf.spin_square()[0],
      np.linalg.eigvalsh(A-B)[0]))
for i, wi in enumerate(pos[:5]):
    print("root %d  %.12f Ha = %.12f eV" % (i, wi, wi * HA2EV))
EOF
```

### Pinned output (2026-07-23)

```
E_UHF = -147.633946785450 Ha  S2≈2.003411  converged=True
sf_channel = alpha_to_beta  n_ov = 27  a_minus_b_min ≈ -0.461612
root 0  0.159296979793 Ha =  4.334691644974 eV
root 1  0.262955422319 Ha =  7.155381562193 eV
root 2  0.262955422319 Ha =  7.155381562193 eV
root 3  0.300659727783 Ha =  8.181367981325 eV
root 4  0.300659727783 Ha =  8.181367981325 eV
```

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `e_tot_ha` (UHF) | **−147.633946785450** |
| `a_minus_b_min` [Ha] | **−0.461612348195** |
| `excitation_ev[0]` | **4.334691644974** |
| `excitation_ev[1]` | **7.155381562193** |
| `excitation_ev[2]` | **7.155381562193** |
| `excitation_ev[3]` | **8.181367981325** |
| `excitation_ev[4]` | **8.181367981325** |

Ha→eV uses **27.211386245988** (CODATA), matching `units::kHartreeInEv`.

Compared with SF-TDA on the same reference (roots 0.045 / 1.568 / 1.568 /
3.038 / 7.849 eV), full SF-RPA shifts the spectrum: the near-zero multiplet
reconstruction and low TDA roots are no longer the lowest positive RPA poles
once the \(B\) coupling is restored under an indefinite \((A-B)\). Degenerate
pairs remain (roots 1–2 and 3–4).

## Tolerance and rationale

**|ω_i − pin_i| < 1e-4 eV** (absolute) for the five lowest positive SF-RPA roots.

Rationale: job gate for this textbook STO-3G case; tight enough to catch
exchange-sign / \(B\)-index / non-Hermitian assembly errors while remaining
honest for residual SCF / ERI / `dgeev` noise. Ground-state E_UHF may be
checked at ~1e-4 Ha as sanity; primary observables are the **excitation
energies [eV]**.

## Literature pointers (method, not numbers)

- Y. Shao, M. Head-Gordon, A. I. Krylov, *J. Chem. Phys.* **118**, 4807 (2003) — SF-TDDFT/CIS.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDHF/RPA response review.
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA (SF-TDA sibling).

## Performance

Fair wall-time vs the offline collinear SF-RPA twin (ROHF warm-start → UHF +
dense SF \(A,B\) non-Hermitian RPA; same geom/basis/method class as
`reg_sf-tddft-o2`):

```bash
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method sf_tddft --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 5 \
  -- ./build/regressions/reg_sf-tddft-o2
```

| Side | Median wall [s] (n=5) |
|------|------------------------|
| PySCF twin (ROHF→UHF + SF-RPA) | **0.0412** |
| Atomiverse `reg_sf-tddft-o2` | **0.392** |
| **ratio** \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **9.51** (≤ 10.0 ✓) |
