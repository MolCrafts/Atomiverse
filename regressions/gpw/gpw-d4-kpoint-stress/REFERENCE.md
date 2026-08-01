# Reference — multi-k GPW PBE-D4 analytic cell stress

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with additive **DFT-D4** (Becke–Johnson two-body, PBE parameters;
ATM off, \(s_9 = 0\)).

**Product cell stress** is the Hellmann–Feynman fixed-\(P_k\) path
(`cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `d4 = true`,
`compute_stress = true`, `stress_use_fd = false`):

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D4}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd4_energy` as the energy path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Sibling twins: `gpw-d4-stress` (Γ dedicated pin), `gpw-kpoint-stress`
(multi-k plain stress), `gpw-d3bj-kpoint-stress` (multi-k D3(BJ) fold structure).

## Why the stress pin is a *difference* of two CP2K runs

DFT-D4 (two-body) is geometry-only — it never enters the Kohn–Sham potential,
so at **every strained cell** the converged density with and without D4 is
identical (for the same electronic k mesh). CP2K's `STRESS_TENSOR NUMERICAL`
re-converges the SCF at each perturbed cell, hence

\[
\sigma^{\mathrm{CP2K}}(\mathrm{D4})-\sigma^{\mathrm{CP2K}}(\mathrm{plain})
 = -\Omega^{-1}\,\partial E_{\mathrm{disp}}/\partial\varepsilon
 \Big|_{\mathrm{CP2K\ FD}},
\]

the pure D4 dispersion stress: electronic stress **and** CP2K's FD
truncation cancel in the difference.

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
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); ATM off |
| CP2K D4 cutoffs | `D4_CUTOFF 30.0` Å (2-body), `R_CUTOFF 20.0` Å (3-body slot); `CALCULATE_C9_TERM .FALSE.`; `FACTOR_S9_TERM 0.0` |
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
| Inputs | `tools/cp2k_oracle/inputs/gpw-d4-kpoint-stress/water_dimer_d4_kpoint_stress.inp` and `water_dimer_plain_kpoint_stress.inp` |
| Note | `tools/cp2k_oracle/inputs/gpw-d4-kpoint-stress/water_dimer_d4_kpoint_stress.note` |
| Wall (fair twin) | **94.525 s** CP2K internal timer on the D4 multi-k NUMERICAL-stress job |
| Date | 2026-08-01 |

### Energies [Hartree]

| Quantity | CP2K (multi-k 1×1×2) | dftd4 3.7.0 |
| -------- | ---- | ---- |
| Total (plain) | −34.33825197808605 | — |
| Total (D4) | −34.33924668432004 | — |
| **Dispersion energy** (printed / standalone) | **−0.00099470623400** | **−0.00099458842745846774** |

ATV D4 is the in-tree `dftd4_energy` path (matches dftd4 3.7.0 to 1e−14 Ha).
CP2K↔dftd4 E_disp residual on this cell is ~1.2×10⁻⁷ Ha (cutoff / CN table
parity under the twin cutoffs above) — hard E_disp gate is therefore the
dftd4 pin; hard \(\Delta\sigma\) gate is the CP2K multi-k difference.

### Numerical stress tensors (`STRESS| Numerical stress tensor [bar]`)

| Component | D4 run | plain run | Δ = D4 − plain [bar] | Δ [Ha/Bohr³] |
| --------- | -------- | --------- | --------------------- | ------------- |
| xx | 5.52041294498E+03 | 5.54281020759E+03 | −22.39726261 | **−7.61267484463E−08** |
| yy | 1.87847935569E+01 | 1.91191382451E+01 | −0.3343446882 | **−1.13641449923E−09** |
| zz | 1.28134653774E+04 | 1.28128742165E+04 | +0.5911609 | **+2.00931506270E−09** |

Conversion: 1 Ha/Bohr³ = 2.9421015697×10⁸ bar (CODATA 2018).

## Dual gate

- **Correctness:** dftd4 \(E_{\mathrm{disp}}\) pin; CP2K multi-k \(\Delta\sigma\);
  product \(\sigma(\mathrm{d4})-\sigma(\mathrm{plain})\) wiring; analytic vs
  multi-SCF FD on all three diagonal axes.
- **Speed:** fair wall uses the **same** CP2K D4 multi-k NUMERICAL-stress
  twin job (1 MPI × 8 OMP, 94.525 s). Product ATV wall is the multi-k
  analytic pass (one multi-k SCF + fixed-\(P_k\) strain evaluations). Target
  \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\).

## Tolerances

| Check | Tol |
| ----- | --- |
| \(E_{\mathrm{disp}}\) vs dftd4 | 1×10⁻⁸ Ha |
| diag \(\sigma_{\mathrm{disp}}\) vs CP2K Δ | 2×10⁻⁹ Ha/Bohr³ |
| dual gate \(\lvert\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\rvert\) | 2×10⁻⁵ Ha/Bohr³ |
| off-diagonal product | 1×10⁻¹² Ha/Bohr³ |
| max \(\lvert\sigma_{\mathrm{diag}}\rvert\) floor | > 1×10⁻¹⁰ Ha/Bohr³ |
| product wall | ≤ 60 s |
| fair wall ratio | ≤ 10 |

## Measured results (2026-08-01, green on rebased tip)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| A \(E_{\mathrm{disp}}\) vs dftd4 | 1e−8 Ha | **2.75e−14** |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **1.36e−09** |
| D wiring vs CP2K multi-k Δσ | 2e−9 Ha/Bohr³ | **2.46e−10** |
| E off-diagonals | exactly 0 | **0** |
| Product wall | ≤ 60 s | **13.11 s** |
| Fair wall ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10 | **0.139** (13.11 / 94.525) |

σ_diag (d4, analytic, 120 Ry): xx −3.613e−05, yy −8.570e−05,
zz −3.081e−06 Ha/Bohr³.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| `oracle` | **cp2k** (multi-k NUMERICAL stress-difference + fair wall) |
| E_disp cross-check | dftd4 3.7.0 (ATV backend) |
| ratio | **0.139** |

## What is checked

1. Product analytic stress with `d4=true`, `k_mesh={1,1,2}` finishes.
2. `e_disp` within 1e-8 Ha of the dftd4 pin.
3. Dual-gate diagonal residual ≤ 2e-5 Ha/Bohr³.
4. \(\sigma(\mathrm{d4})-\sigma(\mathrm{plain})\) matches CP2K multi-k Δσ.
5. Off-diag product ≈ 0; Frame `kStress` unit conversion.
6. Fair wall ratio ≤ 10 vs real CP2K D4 twin.
