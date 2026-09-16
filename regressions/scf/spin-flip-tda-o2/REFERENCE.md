# O₂ UHF/STO-3G collinear spin-flip TDA (SF-CIS) — excitation roots [eV]

## Method

**Collinear spin-flip Tamm–Dancoff** (SF-CIS / SF-TDA) on a high-spin **UHF**
reference (triplet O₂, \(2S=2\)). Ms-decreasing singles \(i_\alpha \to a_\beta\)
(majority→minority) form the response space. With the collinear HF kernel the
Coulomb term vanishes for pure spin-flip and only exchange remains
(Shao, Head-Gordon & Krylov, *J. Chem. Phys.* **118**, 4807 (2003)):

\[
A_{ia,jb}
=
\delta_{ij}\delta_{ab}
(\varepsilon_a^\beta-\varepsilon_i^\alpha)
-(ij|ab)
\]

Dense Hermitian solve \(A X = \omega X\). Public path: `UHF::sf_tda(n_states)`
after a converged `UHF::run`.

PySCF **mainline has no product SF-TDA**; the offline pin builds the same
matrix with PySCF UHF MOs + `ao2mo` (`tools/pyscf_oracle/pin_scf.py sf-tda`).
That script is also the fair-bench twin (`--method sf_tda`).

Observable: lowest SF roots **ω [eV]** (includes a near-zero multiplet
component from Ms-reconstruction of the high-spin reference under mild UHF
spin contamination).

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
- SF-TDA: dense `eigh` of full \(A\); `nstates = 5`

## Oracle: PySCF 2.14.0 integrals (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py sf-tda \
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
no, nv = len(oa), len(vb)
Co, Cv = mo_a[:, oa], mo_b[:, vb]
ijab = ao2mo.general(mol, (Co, Co, Cv, Cv), compact=False).reshape(no, no, nv, nv)
A = np.zeros((no*nv, no*nv))
for i in range(no):
    for a in range(nv):
        p = i*nv + a
        A[p, p] = eb[vb[a]] - ea[oa[i]]
        for j in range(no):
            for b in range(nv):
                A[p, j*nv + b] -= ijab[i, j, a, b]
w = np.linalg.eigvalsh(A)
print("E_UHF = %.12f Ha  S2=%.10f" % (e, mf.spin_square()[0]))
for i, wi in enumerate(w[:5]):
    print("root %d  %.12f Ha = %.12f eV" % (i, wi, wi * HA2EV))
EOF
```

### Pinned output (2026-07-23)

```
E_UHF = -147.633946785450 Ha  S2≈2.003411  converged=True
sf_channel = alpha_to_beta  n_ov = 27
root 0  0.001659121519 Ha =  0.045146996469 eV
root 1  0.057612188548 Ha =  1.567707515056 eV
root 2  0.057612188548 Ha =  1.567707515056 eV
root 3  0.111660939718 Ha =  3.038448959259 eV
root 4  0.288431775262 Ha =  7.848628442263 eV
```

| Quantity | Value |
|----------|-------|
| `pyscf` | 2.14.0 |
| `e_tot_ha` (UHF) | **−147.633946785450** |
| `excitation_ev[0]` | **0.045146996469** |
| `excitation_ev[1]` | **1.567707515056** |
| `excitation_ev[2]` | **1.567707515056** |
| `excitation_ev[3]` | **3.038448959259** |
| `excitation_ev[4]` | **7.848628442263** |

Ha→eV uses **27.211386245988** (CODATA), matching `units::kHartreeInEv`.

Root 0 is the near-zero Ms-reconstruction of the high-spin multiplet (UHF
spin contamination); roots 1–2 are a near-degenerate pair.

## Tolerance and rationale

**|ω_i − pin_i| < 1e-4 eV** (absolute) for the five lowest SF roots.

Rationale: job gate for this textbook STO-3G case; tight enough to catch
exchange-sign / channel-block errors while remaining honest for residual
SCF / ERI noise. Ground-state E_UHF may be checked at ~1e-4 Ha as sanity;
primary observables are the **excitation energies [eV]**.

## Literature pointers (method, not numbers)

- Y. Shao, M. Head-Gordon, A. I. Krylov, *J. Chem. Phys.* **118**, 4807 (2003) — SF-TDDFT/CIS.
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA.
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — response review.

## Performance

Fair wall-time vs the offline collinear SF-CIS twin (ROHF warm-start → UHF +
dense SF-CIS \(A\); same geom/basis/method class as `reg_spin-flip-tda-o2`):

```bash
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method sf_tda --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 5 \
  -- ./build/regressions/reg_spin-flip-tda-o2
```

| Side | Median wall [s] (n=5) |
|------|------------------------|
| PySCF twin (ROHF→UHF + SF-CIS) | **0.0326** |
| Atomiverse `reg_spin-flip-tda-o2` | **0.311** |
| **ratio** \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **9.56** (≤ 10.0 ✓) |

`within_10x = true`. Protocol: median of 5 runs; `OMP_NUM_THREADS=1`.
