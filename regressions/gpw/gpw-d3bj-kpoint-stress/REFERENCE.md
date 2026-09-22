# Reference — multi-k GPW PBE-D3(BJ) analytic cell stress

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with additive **DFT-D3 Becke–Johnson (rational) damping**, PBE
parameters, two-body only (`CALCULATE_C9_TERM .FALSE.`).

**Product cell stress** is the Hellmann–Feynman fixed-\(P_k\) path
(`cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `d3bj = true`,
`compute_stress = true`, `stress_use_fd = false`):

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D3(BJ)}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd3_pbe_bj` as the energy path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Sibling twins: `gpw-d3bj-stress` (Γ dedicated pin), `gpw-kpoint-stress`
(multi-k plain stress), `gpw-mbd-kpoint-stress` (multi-k MBD fold structure).

## Why the stress pin is a *difference* of two CP2K runs

DFT-D3 is geometry-only — it never enters the Kohn–Sham potential, so at
**every strained cell** the converged density with and without D3 is
identical (for the same electronic k mesh). CP2K's `STRESS_TENSOR NUMERICAL`
re-converges the SCF at each perturbed cell, hence

\[
\sigma^{\mathrm{CP2K}}(\mathrm{D3BJ})-\sigma^{\mathrm{CP2K}}(\mathrm{plain})
 = -\Omega^{-1}\,\partial E_{\mathrm{disp}}/\partial\varepsilon
 \Big|_{\mathrm{CP2K\ FD}},
\]

the pure D3(BJ) dispersion stress: electronic stress **and** CP2K's FD
truncation cancel in the difference. Because \(E_{\mathrm{disp}}\) does not
depend on the electronic k mesh, the multi-k Δσ pin is expected to match the
Γ `gpw-d3bj-stress` pin to table precision — the multi-k twin re-runs CP2K
with `KPOINTS` so the claim is not a cross-representation transfer.

## System

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2** (CP2K `FULL_GRID .FALSE.`; ATV TR-folded) |
| Dispersion | DFT-D3(BJ), PBE: \(s_6=1.0\), \(s_8=0.7875\), \(a_1=0.4289\), \(a_2=4.4407\) Bohr; CP2K CN form; ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å = 75.589 Bohr |
| CP2K cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV cutoff | 200 Ry (dispersion gates geometry-only; dual gate grid-consistent) |
| SCF | CP2K `EPS_SCF 1e-8` + `DIAGONALIZATION` (k-points); ATV `tol = 1e-6` |
| Stress step | ATV analytic/FD \(h=2\times10^{-3}\); in-regression D3-only FD \(h=10^{-4}\) |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 1`, `OMP_NUM_THREADS=8`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3bj-kpoint-stress/water_dimer_d3bj_kpoint_stress.inp` and `water_dimer_plain_kpoint_stress.inp` |
| Note | `tools/cp2k_oracle/inputs/gpw-d3bj-kpoint-stress/water_dimer_d3bj_kpoint_stress.note` |
| Wall (fair twin) | **60.111 s** CP2K internal timer on the D3BJ multi-k NUMERICAL-stress job |
| Date | 2026-08-01 |

### Energies [Hartree]

| Quantity | CP2K (multi-k 1×1×2) |
| -------- | ---- |
| Total (plain) | −34.33825197808604 |
| Total (D3BJ) | −34.33962825578086 |
| **Dispersion energy** (printed) | **−0.00137627769481** |

(Total-energy difference reproduces the printed dispersion energy to ~1e−15 Ha.)

### Numerical stress tensors (`STRESS| Numerical stress tensor [bar]`)

| Component | D3BJ run | plain run | Δ = D3BJ − plain [bar] | Δ [Ha/Bohr³] |
| --------- | -------- | --------- | --------------------- | ------------- |
| xx | 5.50901753917E+03 | 5.54281020556E+03 | −33.79266639 | **−1.14858938719E−07** |
| yy | 1.88563460724E+01 | 1.91191402778E+01 | −0.2627942054 | **−8.93219350774E−10** |
| zz | 1.28131582984E+04 | 1.28128742083E+04 | +0.2840901 | **+9.65602625435E−10** |

Conversion: 1 Ha/Bohr³ = 2.9421015697×10⁸ bar (CODATA 2018). Γ-pin residual
vs `gpw-d3bj-stress`: max |Δ| ≈ 3e−14 Ha/Bohr³ (table precision).

## Dual gate

- **Correctness:** CP2K multi-k Δσ + \(E_{\mathrm{disp}}\) pins; product
  \(\sigma(\mathrm{d3bj})-\sigma(\mathrm{plain})\) wiring; analytic vs
  multi-SCF FD on xx; unit twin `GpwStress.D3BjMultiKAnalyticMatchesFd`.
- **Speed:** fair wall uses the **same** CP2K D3BJ multi-k NUMERICAL-stress
  twin job (1 MPI × 8 OMP, 60.111 s). Product ATV wall is the multi-k
  analytic pass (one multi-k SCF + fixed-\(P_k\) strain evaluations). Target
  \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\).

## Tolerances

| Check | Tol |
| ----- | --- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1×10⁻⁸ Ha |
| diag \(\sigma_{\mathrm{disp}}\) vs CP2K Δ | 1×10⁻¹⁰ Ha/Bohr³ |
| product wiring \(\sigma(\mathrm{d3bj})-\sigma(\mathrm{plain})\) | 2×10⁻⁹ Ha/Bohr³ |
| dual gate \(\lvert\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\rvert\) (xx) | 2×10⁻⁵ Ha/Bohr³ |
| off-diagonal product | 1×10⁻⁸ Ha/Bohr³ |
| max \(\lvert\sigma_{\mathrm{diag}}\rvert\) floor | > 1×10⁻¹⁰ Ha/Bohr³ |
| product wall | ≤ 60 s |
| fair wall ratio | ≤ 10 |

## Measured results (2026-08-01, first green)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K (standalone + product) | 1e−8 Ha | **7.6e−12** |
| diag \(\sigma_{\mathrm{disp}}\) vs CP2K multi-k Δ | 1e−10 Ha/Bohr³ | **4.6e−14** (xx), 5.9e−15 (yy/zz) |
| product wiring \(\sigma(\mathrm{d3bj})-\sigma(\mathrm{plain})\) | 2e−9 Ha/Bohr³ | **7.3e−11** |
| dual gate \(\lvert\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\rvert\) (xx) | 2e−5 Ha/Bohr³ | **7.4e−07** |
| off-diagonal product | 1e−8 Ha/Bohr³ | ~0 (pass) |
| multi-k payload | nk ≥ 1, `C_occ_k` | nk = 1 (TR fold of 1×1×2) |
| product wall | ≤ 60 s | **23.0 s** (OMP 8; FD twin 14.6 s) |
| fair wall ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10 | **0.382** (23.0 / 60.111) |

Product analytic diagonals (200 Ry, `tol = 1e−6`, 11 SCF iters,
\(E_{\mathrm{tot}}\) = −34.331707679925 Ha):

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | −2.17974378×10⁻⁵ |
| \(\sigma_{yy}\) | +2.00723524×10⁻⁵ |
| \(\sigma_{zz}\) | +5.82656528×10⁻⁵ |

Unit twin `GpwStress.D3BjMultiKAnalyticMatchesFd`: max |an−fd| **4.7e−11**
Ha/Bohr³; `D3BjMultiKChangesDiagonal` max shift **2.2e−08** Ha/Bohr³.
EOF
