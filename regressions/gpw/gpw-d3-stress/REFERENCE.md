# Reference — periodic water-dimer GPW PBE-D3 (zero damping) analytic cell stress

## Method

Γ-point **GPW** (PBE) with additive **DFT-D3 zero ("original") damping**
(PBE reference parameters, two-body only, `CALCULATE_C9_TERM .FALSE.`) on a
**periodic** water dimer.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → \(P\) and energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[P;W;\varepsilon]\) under orthorhombic diagonal strain at fixed
   fractional coordinates, with geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D3}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd3_pbe` sum as the energy path,
   `kD3Cp2kPairCutoffBohr` = 2 × CP2K `R_CUTOFF 20.0` Å lattice-image cutoff).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `d3=true`,
`compute_stress=true`, `stress_use_fd=false` → Frame `kStress` [eV/Å³];
compare via `last_stress_ha_bohr3()` [Ha/Bohr³].

## Why the stress pin is a *difference* of two CP2K runs

D3 is geometry-only: it adds \(E_{\mathrm{disp}}(\{R\},\mathbf{h})\) to the
total energy and does not enter the Kohn–Sham potential, so the converged
density is identical with and without it. Therefore

\[
\sigma^{\mathrm{CP2K}}(\mathrm{D3}) - \sigma^{\mathrm{CP2K}}(\mathrm{plain})
  = \sigma_{\mathrm{disp}}^{\mathrm{D3}}
\]

isolates exactly the quantity this capability adds, with the (much larger,
grid-noise-limited) electronic stress cancelling — the stress analogue of the
`gpw-d3-forces` force pin. Atomiverse isolates its own dispersion stress the
same way: analytic product with `d3=true` minus analytic product with
`d3=false` (warm-started; same density fixed point).

## System

Same geometry as `gpw-d3-forces` (periodic twin) / `gpw-d4-stress`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D3 zero damping, PBE: \(s_6=1.0\), \(s_8=0.722\), \(s_{r,6}=1.217\), \(s_{r,8}=1.0\); ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å = 75.589 Bohr (periodic — the convention **does** move this pin; see gpw-d3-forces "Cutoff convention") |
| Cutoff | 200 Ry (ATV single grid; CP2K `CUTOFF 200, REL_CUTOFF 40, NGRIDS 4`) — `gpw-d4-stress` cost family. E_disp and Δσ_disp are geometry-only ⇒ cutoff-independent |
| SCF | CP2K `EPS_SCF 1e-7`; Atomiverse `tol = 1e-7` |
| Stress | CP2K `STRESS_TENSOR NUMERICAL`; ATV analytic fixed-P `h=2e-3` |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`, git:5e54ba2) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-stress/water_dimer_d3_stress.inp` and `water_dimer_plain_stress.inp` |
| Wall | **33.6 s** (D3 run), **38.9 s** (plain run) — full NUMERICAL stress (many strained SCFs) |
| Date | 2026-07-31 |

### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain) | −34.338293290245112 |
| Total (D3) | −34.339114184921726 |
| **Dispersion energy** | **−0.00082089467662** |

(Consistent with the periodic D3-zero energy pin −8.20894676620e−04 Ha taken
at CUTOFF 300 for `gpw-d3-forces` — geometry-only, cutoff-independent.)

### Numerical stress tensors (CP2K prints **bar**; diagonal)

| Axis | σ(D3) [bar] | σ(plain) [bar] | Δσ = D3 − plain [bar] |
| ---- | ----------- | -------------- | --------------------- |
| xx | 5.44389060708×10³ | 5.44891104191×10³ | **−5.02043483** |
| yy | −1.06513889194×10² | −1.06247934255×10² | **−0.26595494** |
| zz | 1.28102626350×10⁴ | 1.28091614698×10⁴ | **+1.10116520** |

Unit conversion (CP2K internal table):
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\).

**Hard Δσ pin [Ha/Bohr³]:** xx **−1.70641140246×10⁻⁸**,
yy **−9.03962616420×10⁻¹⁰**, zz **+3.74278507120×10⁻⁹**.

**Soft total σ(D3) pin [Ha/Bohr³]:** xx 1.85034111988×10⁻⁵,
yy −3.62033411836×10⁻⁷, zz 4.35412050331×10⁻⁵.

### Re-run (offline only — never ctest)

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
cd tools/cp2k_oracle/inputs/gpw-d3-stress
OMP_NUM_THREADS=2 mpirun -np 2 cp2k.psmp -i water_dimer_d3_stress.inp -o d3.out
OMP_NUM_THREADS=2 mpirun -np 2 cp2k.psmp -i water_dimer_plain_stress.inp -o plain.out
```

## Tolerances (physical justification)

| Gate | Tol | Why |
| ---- | --- | --- |
| \|E_disp − pin\| | **1×10⁻⁸ Ha** | Same closed-form two-body sum, same published PBE parameters, same 2×R_CUTOFF image convention; `gpw-d3-forces` measured 6.4×10⁻¹¹ agreement. |
| max\|Δσ_atv − Δσ_cp2k\| diag | **2×10⁻⁹ Ha/Bohr³** | Both sides are differences of two runs sharing one density fixed point. CP2K-side noise: NUMERICAL FD with EPS_SCF 1e-7 → ≲10⁻¹¹ Ha/Bohr³ (off-diagonals of the printed tensors, ~10⁻⁶ bar, bound it). ATV-side noise: two independently converged SCFs (tol 1e-7) enter the fixed-P Lagrangian → measured residual below; gate leaves ≥5× headroom over the xx signal 1.7×10⁻⁸ while still 8× above the measured floor. |
| max\|σ_an − σ_fd\| diag (d3=true) | **2×10⁻⁵ Ha/Bohr³** | Internal dual gate, `gpw-d4-stress` precedent: fixed-P Lagrangian vs full re-SCF differ by SCF residual + O(h²). |
| off-diagonal product | **1×10⁻⁸** | Orthorhombic cell: shear identically 0 by construction. |
| max\|σ_diag\| floor | **>1×10⁻¹⁰ Ha/Bohr³** | Stress actually computed, not zero-filled. |
| max\|σ_tot − σ_cp2k\| diag | **1×10⁻⁴ Ha/Bohr³** (sanity) | ATV single 200 Ry grid vs CP2K NGRIDS 4 / REL_CUTOFF 40 differ at the grid-discretisation level (several ×10⁻⁵ Ha/Bohr³ ≈ kbar on this cell — same class as the `gpw-d3-forces` total-force gate being bounded by CP2K's own grid noise). The gate catches unit/sign/blow-up errors only; the decisive stress physics is the Δσ pin above. |

## Atomiverse numbers (first green, 2026-07-31)

`reg_gpw-d3-stress` (OMP 8, epyc login node), 11 SCF iters,
\(E_{\mathrm{tot}}\) = −34.331152296855 Ha:

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \|E_disp − pin\| | 1e−8 Ha | **5.53e−11** (E_disp = −8.20894731919e−04 Ha) |
| diag \|Δσ_atv − Δσ_cp2k\| | 2e−9 Ha/Bohr³ | **2.27e−10** (xx), 2.26e−12 (yy), 1.92e−11 (zz) |
| dual gate max\|σ_an − σ_fd\| diag | 2e−5 Ha/Bohr³ | **7.39e−07** (xx; yy 2.46e−8, zz 1.24e−9) |
| max\|σ_tot − σ_cp2k\| diag (sanity) | 1e−4 Ha/Bohr³ | 4.02e−05 |
| off-diagonal product | 1e−8 | ~0 (pass) |
| max\|σ_diag\| floor | >1e−10 | 5.83e−5 |
| Wall (product + plain twin + all-axes FD twin) | ≤ 60 s | **18.5 + 14.9 + 14.3 = 47.7 s** |

Product analytic diagonals [Ha/Bohr³]: xx −2.16997707×10⁻⁵,
yy +2.00723305×10⁻⁵, zz +5.82684354×10⁻⁵.

The Δσ residual (2.27e−10 on xx) sits at the SCF-noise floor of the
**two-full-products** isolation — both analytic runs re-converge to the same
density fixed point only to `tol = 1e−7`, and the residual difference enters
the fixed-\(P\) Lagrangian FD as δE/(2hΩ). The sibling `gpw-d3bj-stress`
reaches 3.9e−14 because its Δσ side is a strain-FD of the standalone
(no-SCF) dispersion sum; this case deliberately pays the noise to gate the
**product path fold** end-to-end — σ_an(d3) − σ_an(plain) runs
`GPWEvaluator`/`gpw_stress` twice. Signal separation stays decisive:
xx signal 1.71e−8 = 75× the measured residual, tolerance 2e−9 keeps
8.8× headroom.

## Dual gate

- **Correctness:** the six gates above (all green, measured column), plus the
  unit-tier analytic-vs-FD identity `GpwStress.AnalyticMatchesFdWithD3` and
  the `GpwStress.D3ChangesGammaDiagonal` guard (cheap MT H₂) in
  `tests/cpu/gpw/test_cpu_gpw_stress.cpp` (suite green, 11.6 s).
- **Speed:** CP2K NUMERICAL stress twin wall **33.6 s** (np 2 × OMP 2); the
  fair ATV wall is the analytic product (SCF + fixed-P stress) —
  **18.5 s** (OMP 8) → ratio **0.55**, inside the 10× fair wall. The
  regression's extra plain/FD twins (14.9 + 14.3 s) exist only for the
  difference pin and the dual gate, not the product path. Settings caveat:
  cutoffs and core counts differ as stated (ATV single 200 Ry grid, OMP 8 vs
  CP2K NGRIDS 4, 2×2) — both walls deliver the same observable (this cell's
  stress), which is what is compared.

## Notes

- Sibling dispersion-stress products: `regressions/gpw/gpw-d4-stress`
  (D4, dual-gate only — no CP2K DFTD4 build), `gpw-mbd-stress`.
- Sibling D3-zero pins: `gpw-d3-forces` (isolated forces + periodic E_disp),
  `gpw-d3-atm` (isolated, +ATM energy).
- D3(BJ) stress is the separate queue row `gpw-d3bj-stress` (shared fold —
  serialized per queue surface note).
