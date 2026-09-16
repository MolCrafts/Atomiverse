# Reference — periodic water-dimer GPW PBE-D3(BJ) analytic cell stress

## Method

Γ-point **GPW** (PBE) on a **periodic** water dimer with additive **DFT-D3
Becke–Johnson (rational) damping**, PBE parameters, two-body only
(`CALCULATE_C9_TERM .FALSE.`).

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path
(`cpu::gpw::GPWEvaluator` with `d3bj = true`, `compute_stress = true`,
`stress_use_fd = false`):

1. One SCF at the unstrained cell → \(P\) and energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian \(E_L[P;W;\varepsilon]\)
   under orthorhombic diagonal strain at fixed fractional coordinates, with the
   geometry-only \(E_{\mathrm{disp}}^{\mathrm{D3(BJ)}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd3_pbe_bj` as the energy path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Sibling twins: `gpw-d3bj-forces` (same geometry, force pins),
`gpw-d4-stress` (same stress path, D4 — no CP2K stress pin there because the
Arrhenius CP2K build lacks libdftd4; **this case carries the real oracle
stress pin** for the dispersion stress fold), `gpw-pbe-d3bj-ice` (D3(BJ)
energy pin).

## Why the stress pin is a *difference* of two CP2K runs

DFT-D3 is geometry-only — it never enters the Kohn–Sham potential, so at
**every strained cell** the converged density with and without D3 is
identical, and \(E^{\mathrm{D3BJ}}(\varepsilon)=E^{\mathrm{plain}}(\varepsilon)
+E_{\mathrm{disp}}(\varepsilon)\) exactly. CP2K's `STRESS_TENSOR NUMERICAL`
re-converges the SCF at each perturbed cell, hence

\[
\sigma^{\mathrm{CP2K}}(\mathrm{D3BJ})-\sigma^{\mathrm{CP2K}}(\mathrm{plain})
 = -\Omega^{-1}\,\partial E_{\mathrm{disp}}/\partial\varepsilon
 \Big|_{\mathrm{CP2K\ FD}},
\]

the pure D3(BJ) dispersion stress: the (three-orders-larger) electronic
stress **and** CP2K's own FD truncation cancel in the difference. This
isolates exactly the quantity this capability adds — the same construction as
the `gpw-d3bj-forces` force pin.

## System

Verbatim the shipped `gpw-d3bj-forces` twin geometry:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | DFT-D3(BJ), PBE: \(s_6=1.0\), \(s_8=0.7875\), \(a_1=0.4289\), \(a_2=4.4407\) Bohr; CP2K CN form; ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å = 75.589 Bohr |
| CP2K cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV cutoff | 200 Ry (both dispersion gates are geometry-only ⇒ cutoff-independent; the dual gate is grid-consistent by construction) |
| SCF | CP2K `EPS_SCF 1e-8` (tightened vs the 1e-7 forces twin — lowers the FD-difference noise floor); ATV `tol = 1e-6` |
| Stress step | ATV analytic/FD \(h=2\times10^{-3}\); in-regression D3-only FD \(h=10^{-4}\); CP2K numerical-stress default cell-perturbation FD |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3bj-stress/water_dimer_d3bj_stress.inp` and `water_dimer_plain_stress.inp` |
| Wall | 83.8 s (D3BJ run), 64.5 s (plain run) — CP2K internal timer |
| Date | 2026-07-31 |

### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain, periodic) | −34.338251977942591 |
| Total (D3BJ, periodic) | −34.339628255637400 |
| **Dispersion energy** (printed) | **−0.00137627769481** |

(The total-energy difference reproduces the printed dispersion energy to
1.4e−16 Ha.)

### Numerical stress tensors (`STRESS| Numerical stress tensor [bar]`)

| Component | D3BJ run | plain run | Δ = D3BJ − plain [bar] | Δ [Ha/Bohr³] |
| --------- | -------- | --------- | --------------------- | ------------- |
| xx | 5.50901740298E+03 | 5.54281007140E+03 | −33.79266842 | **−1.1485894562E−07** |
| yy | 1.88563460724E+01 | 1.91191341799E+01 | −0.26278811 | **−8.9319862445E−10** |
| zz | 1.28131586907E+04 | 1.28128745905E+04 | +0.28410020 | **+9.6563695465E−10** |

Conversion: 1 Ha/Bohr³ = 2.9421015697×10⁸ bar (CODATA 2018). The dimer axis
is x, so \(\Delta\sigma_{xx}\) carries the ~1.1e−7 signal; yy/zz are the
lattice-image / H-offset residuals (~1e−9), still ~30× above the pin floor.

## Measured results (2026-07-31, this pin)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K (standalone + product) | 1e−8 Ha | **7.6e−12** |
| diag \(\sigma_{\mathrm{disp}}\) vs CP2K Δ | 1e−10 Ha/Bohr³ | **3.9e−14** (xx), 2.7e−14 (yy), 3.2e−14 (zz) |
| dual gate \(|\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}|\) (xx in e2e) | 2e−5 Ha/Bohr³ | **7.4e−07** (pinning ran all axes: yy 2.5e−8, zz 1.2e−9) |
| off-diagonal product | 1e−8 Ha/Bohr³ | ~0 (pass) |
| max \(|\sigma_{\mathrm{diag}}|\) floor | > 1e−10 Ha/Bohr³ | 5.83e−5 |
| Wall (product + x-only FD twin) | ≤ 60 s | **28.4 s + 4.8 s** (warm-started FD; wave-1 loaded node — 3-axis cold-measure pinning FD was 79.4 s) |

Product analytic diagonals (first green, 200 Ry, `tol = 1e−6`, 11 SCF iters,
\(E_{\mathrm{tot}}\) = −34.331707679811 Ha):

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | −2.17975632×10⁻⁵ |
| \(\sigma_{yy}\) | +2.00723385×10⁻⁵ |
| \(\sigma_{zz}\) | +5.82656483×10⁻⁵ |

The e2e FD twin re-SCFs the **x axis only** (dimer axis — carries the
1.15e−7 dispersion-stress signal **and** the largest measured analytic↔FD
residual) to hold the 60 s budget; the all-axes identity is the cheap MT H₂
unit twin `GpwStress.AnalyticMatchesFdWithD3Bj`.

## Tolerance justification

- **\(E_{\mathrm{disp}}\) 1e−8 Ha:** closed-form two-body sum with published
  PBE-BJ parameters and the CP2K CN convention; same pin class as
  `gpw-d3bj-forces` (measured 8.9e−11 there). Only table transcription can
  move it.
- **\(\sigma_{\mathrm{disp}}\) 1e−10 Ha/Bohr³:** the pin is a difference of
  two independently converged CP2K numerical-stress runs at `EPS_SCF 1e−8`,
  so it inherits (SCF energy noise)/(h·Ω) ≲ 1e−11 Ha/Bohr³ plus the C6/CN
  table parity; measured 3.9e−14 against the 1.15e−7 xx signal — the gate
  keeps ~1000× separation from the signal and even holds the tiny yy/zz
  components (~9e−10) to ~30× above tolerance floor.
- **Dual gate 2e−5 Ha/Bohr³:** same bound as `gpw-d4-stress` (measured
  7.4e−7 there): the analytic fixed-P Lagrangian vs six fully re-converged
  SCFs at `tol = 1e−6` — the residual is SCF-convergence noise, not the
  dispersion fold.

## Dual gate

- **Correctness:** the pins above, plus the decisive analytic-vs-multi-SCF-FD
  identity on the same `d3bj = true` method in this binary, plus the cheap MT
  H₂ unit twin `GpwStress.AnalyticMatchesFdWithD3Bj`
  (`tests/cpu/gpw/test_cpu_gpw_stress.cpp`).
- **Speed:** CP2K numerical-stress D3BJ twin 83.8 s (18 perturbed SCFs +
  reference, 2 MPI × 2 OMP, 300 Ry); Atomiverse product analytic pass
  **31.1 s** (one SCF + 7 fixed-P grid evaluations, OMP 8, 200 Ry, wave-1
  loaded login node) → ratio ≈ **0.37**, inside the 10× fair wall.
  Settings caveat: the cutoffs and core counts differ as stated — both
  produce the same deliverable (the cell stress of this system), which is
  the wall being compared.

## Notes

- The FD-twin wall in this binary (multi-SCF `gpw_stress_fd`) is gate
  overhead, not the product path; the product path is the analytic pass.
- Siblings: `gpw-d3-stress` (zero damping, same fold) is a separate queue row;
  the fold itself (`energy_fixed_P` / `energy_fixed_P_k` in
  `src/cpu/gpw/internal/gpw_stress.cpp`) predates this pin — this case adds
  the missing oracle evidence, the unit twin, and the e2e product proof.
