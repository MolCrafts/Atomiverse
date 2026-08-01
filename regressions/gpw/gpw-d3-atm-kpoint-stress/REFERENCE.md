# Reference — multi-k GPW PBE-D3 three-body ATM (C9) analytic cell stress

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh **1×1×2**) on the
**periodic 6 Å** water-dimer cell of `gpw-d3-atm-stress` /
`gpw-d3-atm-forces`, with Grimme **DFT-D3** (zero damping, PBE) including the
**Axilrod–Teller–Muto three-body C9** term (`CALCULATE_C9_TERM .TRUE.`)
folded into the **cell stress**.

**Product cell stress** is the Hellmann–Feynman fixed-\(P_k\) path
(`cpu::gpw::GPWEvaluator` with `d3 = true`, `d3_atm = true`,
`k_mesh = {1,1,2}`, `compute_stress = true`, `stress_use_fd = false`):

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\) and energy.
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D3+ATM}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (`dftd3_pbe(..., calculate_c9 = true)` inside
   `energy_fixed_P_k` — the same call as the Γ path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Configurable **`d3_rcut_bohr`** (CP2K `&PAIR_POTENTIAL R_CUTOFF`, doubled
convention): this twin uses `R_CUTOFF 10.0` → 37.79 Bohr so the product folds
exactly the oracle's functional.

Sibling twins: `gpw-d3-atm-stress` (Γ ATM stress pin — same geometry),
`gpw-kpoint-stress` (multi-k electronic stress, no dispersion),
`gpw-mbd-kpoint-stress` (multi-k MBD stress fold pattern).

## Why the pin is a *difference* of two CP2K runs

DFT-D3 — with or without C9 — is geometry-only and never enters the
Kohn–Sham potential, so at **every strained cell** the converged density is
identical with `CALCULATE_C9_TERM` `.TRUE.` and `.FALSE.`, and the two-body
dispersion energy is identical too. CP2K's `STRESS_TENSOR NUMERICAL`
re-converges the multi-k SCF at each perturbed cell, hence

\[
\sigma^{\mathrm{CP2K}}(\mathrm{C9})-\sigma^{\mathrm{CP2K}}(\mathrm{2b})
 = -\Omega^{-1}\,\partial E_{\mathrm{ATM}}/\partial\varepsilon
 \Big|_{\mathrm{CP2K\ FD}},
\]

the pure three-body ATM stress: the electronic stress, the two-body
dispersion stress, **and** the shared parts of CP2K's FD scheme cancel.
Because ATM is geometry-only, the multi-k Δσ matches the Γ pin to
≲ 2×10⁻¹³ Ha/Bohr³ — the k mesh does not enter the dispersion functional.

## System

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **6 × 6 × 6 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2** (CP2K DIAGONALIZATION; OT is Γ-only) |
| Dispersion | DFT-D3 zero damping, PBE; ATM per run (`CALCULATE_C9_TERM` T/F) |
| Pair/ABC cutoff | `R_CUTOFF 10.0` Å → doubled 37.794 Bohr (`d3_rcut_bohr`) |
| CP2K cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV cutoff | 150 Ry (dispersion gates are geometry-only ⇒ cutoff-independent) |
| SCF | CP2K `EPS_SCF 1e-8` + DIRECT_P_MIXING; ATV `tol = 1e-6` |
| Stress step | ATV analytic/FD \(h=2\times10^{-3}\); ATM-only FD at CP2K `DX 0.001` Bohr |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Host | Arrhenius epyc login (`arrhenius1`), offline only |
| Binary | `/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/bin/cp2k.psmp` |
| Env | `source …/cp2k/install/env.sh`; `CP2K_DATA_DIR` set by env |
| Fair wall job | `mpirun -np 1`, `OMP_NUM_THREADS=8` |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-atm-kpoint-stress/water_dimer_d3_atm_k_stress.inp` (C9 T) and `water_dimer_d3_k_stress.inp` (C9 F) |
| Output lines | final `Dispersion energy:` + `STRESS\| Numerical stress tensor [bar]` table |
| Wall (C9, fair) | **18.74 s** (`/usr/bin/time` wall_sec; 1 MPI × 8 OMP) |
| Wall (2b) | 20.23 s (companion; not used for ratio) |
| Date | 2026-08-01 |

### Energies [Hartree]

| Quantity | CP2K multi-k |
| -------- | ------------ |
| Total (C9) | −34.34639787446541 |
| **E_disp (C9)** (printed) | **−0.00195809486938** |
| **E_disp (two-body)** (printed) | **−0.00195887984277** |

E_disp values are bit-identical to the Γ `gpw-d3-atm-stress` pins
(geometry-only). ATM contribution +7.8497339e−7 Ha.

### Numerical stress tensors (`STRESS| Numerical stress tensor [bar]`)

| Component | C9 run | two-body run | Δ = C9 − 2b [bar] | Δ [Ha/Bohr³] |
| --------- | ------ | ------------ | ----------------- | ------------- |
| xx | 4.73259589239E+04 | 4.73310041954E+04 | −5.04527150 | **−1.714852930966E−08** |
| yy | 2.31473723364E+03 | 2.30198827547E+03 | +12.74895817 | **+4.333282814332E−08** |
| zz | 9.99712094855E+04 | 9.99685345898E+04 | +2.67489570 | **+9.091785707051E−09** |

Conversion: 1 Ha/Bohr³ = 2.9421015697×10⁸ bar (CODATA 2018). Off-diagonals
of both tables are ≤ 2e−5 bar (≈ 7e−14 Ha/Bohr³). Residual vs the Γ ATM
stress pin: max |Δσ_multi-k − Δσ_Γ| ≈ 1.9×10⁻¹³ Ha/Bohr³ (xx).

## Measured results (2026-08-01, first green; `run_reg` target, OMP 8)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| E_disp (C9) vs CP2K (standalone) | 1e−9 Ha | **2.79e−10** |
| E_disp (two-body) vs CP2K | 1e−9 Ha | **2.83e−10** |
| E_disp (C9) product multi-k | 1e−9 Ha | **1.18e−10** |
| diag σ_ATM vs multi-k CP2K Δ | 5e−12 Ha/Bohr³ | **8.11e−14** (xx; yy 1.1e−14, zz 3.8e−14) |
| ATM engage floor max \|σ_ATM\| | ≥ 1e−9 Ha/Bohr³ | 4.33e−8 |
| dual gate \|σ_an − σ_fd\| (xx) | 2e−5 Ha/Bohr³ | **6.26e−09** |
| multi-k payload | n_kpts ≥ 1, C_occ_k | nk=1 (TR-folded 1×1×2), C_occ_k present |
| Wall product | ≤ 60 s | **5.34 s** (+ 2.09 s x-only FD twin; 9 SCF iters) |
| Fair ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10 | **0.285** (\(T_{\mathrm{atv}}=5.34\) s / \(T_{\mathrm{cp2k}}=18.74\) s) |

Product analytic diagonals (150 Ry, multi-k, `tol = 1e−6`,
\(E_{\mathrm{tot}}\) = −34.301731577668 Ha):

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | +7.06654524×10⁻⁵ |
| \(\sigma_{yy}\) | −1.57596903×10⁻⁴ |
| \(\sigma_{zz}\) | +2.23103782×10⁻⁴ |

## The ATM strain-FD is step-locked (re-pin warning)

Same as `gpw-d3-atm-stress`: the ATM triple enumeration passes **hard**
gates, so the FD window must match CP2K's `DX 0.001` Bohr. Any re-pin must
use the oracle's `DX`.

## Dual gate

- **Correctness:** the pins above, plus analytic-vs-multi-SCF-FD identity on
  the same multi-k `d3 + d3_atm` method in this binary (x axis), plus the
  cheap unit twin `GpwStress.D3AtmMultiKAnalyticMatchesFd`
  (`tests/cpu/gpw/test_cpu_gpw_stress.cpp`) — periodic H₂, mesh 1×1×2,
  short `d3_rcut_bohr`.
- **Speed:** CP2K multi-k numerical-stress C9 twin **18.74 s** (1 MPI × 8
  OMP, 300 Ry); Atomiverse product multi-k analytic pass (one SCF + fixed-P_k
  grid evaluations, OMP 8, 150 Ry). Ratio must be ≤ 10.

## Notes

- ATM still contributes **no** `D3Result::virial` entry; the fixed-P_k path
  rebuilds dispersion energy under strain (the fold this pin certifies).
- Sibling multi-k dispersion stress pins: `gpw-d3-kpoint-stress` (two-body
  zero), `gpw-d3bj-kpoint-stress`, `gpw-d4-kpoint-stress`, etc.
- CP2K uses DIAGONALIZATION under `&KPOINTS` (OT is Γ-only); ATV multi-k
  SCF uses its standard k-path DIIS/mixing.
