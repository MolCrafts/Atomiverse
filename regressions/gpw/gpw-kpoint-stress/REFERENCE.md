# Reference — GPW analytic cell stress at Monkhorst–Pack k-points (diamond Si)

## Method

Closed-shell GPW Kohn–Sham DFT (PBE) with **Monkhorst–Pack k-point sampling**
and **Hellmann–Feynman fixed-\(P_k\) analytic cell stress**:

1. One SCF at the unstrained cell → occupied MOs `C_occ_k` / band energies
   `eps_k` → rebuild \(P_k\), \(W_k\) per k.
2. Central finite difference of the **non-self-consistent** Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates (rebuild \(S_k,T_k,V_{\mathrm{nl}}^k\), grid,
   Poisson, XC; do **not** re-converge SCF).
3. \(\sigma_{aa} = -\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same mesh.

## System

Silicon diamond in the **body-centred-tetragonal 4-atom cell** (same twin
family as `gpw-kpoints-si`):

| | |
| --- | --- |
| Lattice | a = 5.43 Å; cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → n_occ = 8 |
| XC | PBE |
| Density cutoff | 100 Ry |
| k mesh | MONKHORST-PACK 1×1×2, time-reversal folded → 1 k |
| SCF | DIIS, tol 1e-8 Ha, EPS_DEFAULT 1e-7 |
| Stress step | h = 2×10⁻³ (dimensionless) |

## Offline oracle status

**No CP2K binary / module on the worker host at pin time** (2026-07-25) —
same status as `gpw-kpoints-si`. Absolute CP2K stress digits are **not
invented**. Pin strategy:

1. **Internal dual-gate** (decisive): analytic fixed-\(P_k\) vs multi-SCF
   strain-FD, max diag \(\le 10^{-4}\) Ha/Bohr³.
2. **Energy band** vs the supercell-folding pin of `gpw-kpoints-si` A2
   (mesh 1×1×2 at 95 Ry: \(E = -15.3483059090\) Ha) with a soft 5×10⁻³ Ha
   band at 100 Ry.
3. Replace with a cited CP2K `STRESS_TENSOR NUMERICAL` / `ANALYTICAL` number
   by running `tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp` when offline
   CP2K exists.


## Atomiverse numbers (first green, 2026-07-25)

| Quantity | Value |
| --- | --- |
| **E(mesh 1×1×2 TR, 100 Ry)** | **−15.3483039265 Ha** (SCF 7 iters) |
| σ_xx analytic | 8.83135454e-04 Ha/Bohr³ |
| σ_yy analytic | 8.83135454e-04 Ha/Bohr³ |
| σ_zz analytic | 2.60970168e-04 Ha/Bohr³ |
| max\|σ_an−σ_fd\| | **7.2e-08** Ha/Bohr³ |
| T_atv (OMP=8, Release, dual-gate) | ~171 s |

## Tolerances

| Check | Tol |
| ----- | --- |
| max\|\σ_an − σ_fd\| diag | 1×10⁻⁴ Ha/Bohr³ |
| off-diagonal product | 1×10⁻⁸ |
| \|E − E_fold pin\| soft | 5×10⁻³ Ha |
| max\|\σ_diag\| floor | > 1×10⁻⁶ Ha/Bohr³ |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary/module on the pin host for a fair wall twin. |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |

When CP2K is available, run the twin single-rank, record both walls, and
replace this section (target r ≤ 10).

## Provenance

- Spec: `.claude/specs/gpw-kpoint-stress/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp`
- Deps: `gpw-kpoints-si` ✓, `gpw-analytic-stress` ✓
