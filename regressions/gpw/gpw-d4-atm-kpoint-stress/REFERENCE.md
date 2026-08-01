# Reference — multi-k GPW PBE-D4 three-body ATM (\(s_9\)) analytic cell stress

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh **1×1×2**) on a periodic
water dimer with additive **DFT-D4** including the Axilrod–Teller–Muto
**three-body** term (\(s_9 = 1\), dftd4 `pbe-D4-ATM`).

**Product cell stress** is the Hellmann–Feynman fixed-\(P_k\) path
(`cpu::gpw::GPWEvaluator` with `d4 = true`, `d4_atm = true`,
`k_mesh = {1,1,2}`, `compute_stress = true`, `stress_use_fd = false`):

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D4+ATM}}(\{\mathbf{R}\},\mathbf{h})\)
   (**s9 = 1**) rebuilt at each strained cell (same `dftd4_energy` as the
   energy path / Γ `gpw-d4-atm-stress`).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Sibling twins: `gpw-d4-atm-stress` (Γ ATM stress), `gpw-d4-kpoint-stress`
(multi-k two-body D4 stress), `gpw-d3-atm-kpoint-stress` (multi-k D3 ATM),
`gpw-d4-atm-kpoint-forces` (multi-k D4 ATM forces).

## Why the stress pin is a *difference* of two CP2K runs

DFT-D4 — with or without ATM — is geometry-only and never enters the
Kohn–Sham potential, so at **every strained cell** the converged density with
`FACTOR_S9_TERM 1.0` and `0.0` is identical (same electronic k mesh). CP2K's
`STRESS_TENSOR NUMERICAL` re-converges the multi-k SCF at each perturbed cell,
hence

\[
\sigma^{\mathrm{CP2K}}(s_9{=}1)-\sigma^{\mathrm{CP2K}}(s_9{=}0)
 = -\Omega^{-1}\,\partial E_{\mathrm{ATM}}/\partial\varepsilon
 \Big|_{\mathrm{CP2K\ FD}},
\]

the pure three-body ATM stress: electronic stress, two-body D4 stress, **and**
shared FD truncation cancel. Because ATM is geometry-only, multi-k Δσ matches
the Γ dftd4-derived ATM stress to ≲ 2×10⁻¹¹ Ha/Bohr³.

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
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); **ATM on**: \(s_9=1\) |
| CP2K D4 cutoffs | `D4_CUTOFF 30.0` Å, `R_CUTOFF 20.0` Å; ATM twin: `CALCULATE_C9_TERM .TRUE.`, `FACTOR_S9_TERM 1.0`; 2b twin: C9 F, s9 0 |
| CP2K cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV cutoff | 120 Ry (dispersion gates geometry-only; dual gate grid-consistent) |
| SCF | CP2K `EPS_SCF 1e-8` + `DIAGONALIZATION` (k-points); ATV `tol = 1e-6` |
| Stress step | ATV analytic/FD \(h=2\times10^{-3}\) |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) **with `libdftd4`** |
| Binary | private Arrhenius rebuild `/nobackup/.../cp2k-dftd4/install` (`cp2kflags: … libdftd4`); stock source install lacks DFTD4 |
| Host | Arrhenius epyc login, offline only (`mpirun -np 1`, `OMP_NUM_THREADS=8`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d4-atm-kpoint-stress/water_dimer_d4_atm_kpoint_stress.inp` and `water_dimer_d4_2b_kpoint_stress.inp` |
| Note | `tools/cp2k_oracle/inputs/gpw-d4-atm-kpoint-stress/water_dimer_d4_atm_kpoint_stress.note` |
| Wall (fair twin) | **137.945 s** CP2K internal timer on the **ATM** multi-k NUMERICAL-stress job |
| Date | 2026-08-01 |

### Energies [Hartree]

| Quantity | CP2K multi-k | dftd4 3.7.0 |
| -------- | ------------ | ----------- |
| Total (ATM) | −34.33924765693281 | — |
| Total (2b) | −34.33924668432004 | — |
| **E_disp (ATM)** | −0.00099567884677 | **−0.0009955447583424029** |
| **E_disp (2b)** | −0.00099470623400 | −0.00099458842745846774 |

ATV D4 is the in-tree `dftd4_energy` path (matches dftd4 3.7.0 to ~1e−14 Ha on
the same cutoffs). CP2K↔dftd4 E_disp residual on this cell is ~1.3×10⁻⁷ Ha
(cutoff / CN table parity under the twin cutoffs) — hard E_disp gate is
therefore the **dftd4** pin; hard \(\Delta\sigma\) gate is the CP2K multi-k
difference.

### Numerical stress tensors (`STRESS| Numerical stress tensor [bar]`)

| Component | ATM run | 2b run | Δ = ATM − 2b [bar] | Δ [Ha/Bohr³] |
| --------- | ------- | ------ | ------------------- | ------------- |
| xx | 5.52018710595E+03 | 5.52041294904E+03 | −0.22584309 | **−7.67625062051E−10** |
| yy | 1.89618271881E+01 | 1.87847915243E+01 | +0.1770356638 | **+6.01731991931E−10** |
| zz | 1.28133914386E+04 | 1.28134653693E+04 | −0.07393070 | **−2.51285342293E−10** |

Conversion: 1 Ha/Bohr³ = 2.9421015697×10⁸ bar (CODATA 2018).

### dftd4 strained-cell ATM stress (k-independent cross-check)

Same pins as `gpw-d4-atm-stress` (`tools/cp2k_oracle/inputs/gpw-d4-atm/pin_strain_*`):
σ^ATM_xx ≈ −7.47×10⁻¹⁰, yy ≈ +5.87×10⁻¹⁰, zz ≈ −2.45×10⁻¹⁰ Ha/Bohr³.
CP2K multi-k Δσ residual vs that FD is ≲ 2×10⁻¹¹ Ha/Bohr³.

## Dual gate

- **Correctness:** dftd4 base + strained-cell \(E_{\mathrm{disp}}\) pins;
  CP2K multi-k \(\Delta\sigma\); product \(\sigma(\mathrm{ATM})-\sigma(\mathrm{2b})\)
  wiring; analytic vs multi-SCF FD on the xx axis of the same multi-k
  `d4 + d4_atm` method. Cheap unit twin:
  `GpwStress.D4AtmMultiKAnalyticMatchesFd` (periodic H₂, mesh 1×1×2).
- **Speed:** fair wall uses the **same** CP2K ATM multi-k NUMERICAL-stress twin
  job (1 MPI × 8 OMP, 137.945 s). Product ATV wall is the multi-k analytic
  pass (one multi-k SCF + fixed-\(P_k\) strain evaluations). Target
  \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\).

## Tolerances

| Check | Tol |
| ----- | --- |
| strained \(\lvert E_{\mathrm{disp}}-\mathrm{pin}\rvert\) (s9=1 and 2b, 12 values) | 1×10⁻⁹ Ha |
| \(\lvert\sigma^{\mathrm{ATM}}_{\mathrm{standalone}}-\mathrm{dftd4\ FD}\rvert\) | 1×10⁻¹² Ha/Bohr³ |
| \(\lvert\sigma^{\mathrm{ATM}}-\mathrm{CP2K}\ \Delta\sigma\rvert\) | 5×10⁻¹¹ Ha/Bohr³ |
| ATM engage floor max \(\lvert\sigma^{\mathrm{ATM}}\rvert\) | ≥ 5×10⁻¹⁰ Ha/Bohr³ |
| product \(e_{\mathrm{disp}}\) vs dftd4 ATM base | 1×10⁻⁸ Ha |
| dual gate \(\lvert\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\rvert\) (xx) | 2×10⁻⁵ Ha/Bohr³ |
| product wiring \(\lvert\Delta\sigma_{\mathrm{xx}}-\sigma^{\mathrm{ATM}}_{\mathrm{xx}}\rvert\) | 3×10⁻¹⁰ Ha/Bohr³ |
| off-diagonal product | 1×10⁻¹² Ha/Bohr³ |
| product wall | ≤ 60 s |
| fair wall ratio | ≤ 10 |

## Measured results (2026-08-01, green `run_reg`, OMP 8)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| strained E_disp parity (12 values) | 1e−9 Ha | **3.83e−12** |
| σ_ATM vs CP2K multi-k Δ | 5e−11 Ha/Bohr³ | **2.02e−11** (xx; yy 1.4e−11, zz 6.7e−12) |
| dual gate \|σ_an − σ_fd\| xx | 2e−5 Ha/Bohr³ | **1.36e−09** |
| wiring product Δσ_xx | 3e−10 Ha/Bohr³ | **4.00e−12** |
| E_disp product vs dftd4 ATM | 1e−8 Ha | **2.76e−14** |
| multi-k payload | n_kpts ≥ 1 | nk=1 (TR-folded 1×1×2), 11 SCF iters |
| Product wall | ≤ 60 s | **10.58 s** (+ 7.04 s x-only FD twin) |
| Fair wall ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10 | **0.077** (10.58 / 137.945) |

Product analytic diagonals (120 Ry, multi-k, `tol = 1e−6`):

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | −3.61281682×10⁻⁵ |
| \(\sigma_{yy}\) | −8.57001176×10⁻⁵ |
| \(\sigma_{zz}\) | −3.08105120×10⁻⁶ |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| `oracle` | **cp2k** (multi-k NUMERICAL stress-difference ATM−2b + fair wall) |
| E_disp / strained cross-check | dftd4 3.7.0 (ATV backend) |
| ratio | **0.077** |

## What is checked

1. Standalone `dftd4_energy(s9=1/0)` at all 6 strained cells matches the
   offline dftd4 printed energies; derived σ^ATM is non-vacuous and matches
   the multi-k CP2K Δσ pin.
2. Product multi-k analytic stress with `d4 + d4_atm`, `k_mesh={1,1,2}`
   finishes; `e_disp` within 1e-8 Ha of the dftd4 ATM base pin.
3. Dual-gate residual (xx) ≤ 2e-5 Ha/Bohr³ vs multi-SCF strain-FD.
4. Product \(\sigma(\mathrm{ATM})-\sigma(\mathrm{2b})\) equals standalone
   σ^ATM (s9 fold inside fixed-\(P_k\) rebuild).
5. Off-diag product ≈ 0; Frame `kStress` unit conversion.
6. Fair wall ratio ≤ 10 vs real CP2K ATM twin.
