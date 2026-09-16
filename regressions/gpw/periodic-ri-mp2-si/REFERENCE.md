# Reference — Γ-point RI-GPW MP2 product path on periodic Si₂

## Method

Product path for **periodic closed-shell MP2** on top of a converged
**Γ-point GPW PBE** SCF. The occupied–virtual MO repulsion integrals
\((ia|jb)\) are assembled the **RI-GPW** way (Del Ben, Hutter,
VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012); CP2K
`&WF_CORRELATION` `METHOD MP2_GPW` + `&WFC_GPW`):

1. pair density \(\rho_{ia}=\psi_i\psi_a\) on the SCF density grid;
2. plane-wave Coulomb intermediate: FFT → \(V_{ia}(\mathbf G)=4\pi
   \tilde\rho_{ia}/|\mathbf G|^2\) (G=0 dropped — neutral pairs) → IFFT
   → store three-index \(V_{ia}(\mathbf r)\)
   (\(\mathcal O(n_{\mathrm{ov}} N_g)\) memory);
3. \((ia|jb)=\sum_g w\, V_{ia}(g)\,\rho_{jb}(g)\);
4. canonical closed-shell doubles sum with OS/SS decomposition.

Numerically identical to the Poisson foothold path (`mp2_ri = false`,
`periodic-mp2-si`) — same kernel, product intermediate layout. Gaussian
cFIT / `RI_AUX` tables and multi-k MP2 are residuals.

```cpp
cpu::gpw::GpwDft m;
// ... members; build(); run(atoms);
m.mp2_ri = true;                     // product default
const cpu::gpw::Mp2Result& r = m.mp2(atoms);
// r.ri == true; r.e_corr [Ha]
```

## System

| Item | Value |
|------|--------|
| Crystal | artificial sc-Si₂ (same as `periodic-mp2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, Γ only |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE** |
| Cutoff | **30 Ry** (SCF + WFC_GPW) |
| SCF | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| MP2 | all kept virtuals; `mp2_ri = true` |

## Offline CP2K pin (2026-07-25)

| Item | Value |
|------|--------|
| Binary | CP2K **2026.1** (Development Version), `cp2k.psmp` |
| Input | `tools/cp2k_oracle/inputs/periodic-ri-mp2-si.inp` |
| Host | arrhenius1 (EPYC), OMP=4, 1 MPI rank |
| E_SCF (PBE) | **−7.595338517019** Ha |
| E_corr (SO) | −0.237578621868 Ha |
| E_corr (SS) | −0.038509118792 Ha |
| **E_corr (total)** | **−0.276087740659** Ha |
| E_total | −7.871426257678 Ha |
| Wall (total) | **576.7 s** (SCF ~3.5 min + MP2 ~369 s) |

## Atomiverse gates

| Quantity | Pin / rule | Tol |
|----------|------------|-----|
| E_corr (RI-GPW) | **−0.276087740659** Ha (CP2K) | **1e-3 Ha** |
| \|E_corr(RI) − E_corr(Poisson)\| | 0 (same kernel) | **1e-10 Ha** |
| OS+SS ≡ E_corr; E_total ≡ E_SCF+E_corr | exact | 1e-12 |
| `Mp2Result::ri` | true on product path | — |

Hard gates: SCF converged; E_corr finite and **strictly negative**.

## Perf

| Engine | Wall (this deck, OMP=4) |
|--------|-------------------------|
| CP2K 2026.1 | 576.7 s |
| Atomiverse (SCF + RI MP2 + can cross-check) | ~16–30 s (typical) |

\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \ll 1\) on this host → **within_10x**.
Never launch CP2K from ctest.
