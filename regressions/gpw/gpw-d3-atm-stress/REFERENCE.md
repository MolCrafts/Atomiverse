# Reference — GPW PBE-D3 three-body ATM (C9) analytic cell stress

## Method

Γ-point **GPW** (PBE) on the **periodic 6 Å** water-dimer cell of
`gpw-d3-atm-forces`, with Grimme **DFT-D3** (zero damping, PBE) including the
**Axilrod–Teller–Muto three-body C9** term (`CALCULATE_C9_TERM .TRUE.`)
folded into the **cell stress**.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path
(`cpu::gpw::GPWEvaluator` with `d3 = true`, `d3_atm = true`,
`compute_stress = true`, `stress_use_fd = false`):

1. One SCF at the unstrained cell → \(P\) and energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian \(E_L[P;W;\varepsilon]\)
   under orthorhombic diagonal strain at fixed fractional coordinates, with
   the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D3+ATM}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (`dftd3_pbe(..., calculate_c9 = true)`, the
   same call as the energy path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

This regression also ships the configurable **`d3_rcut_bohr`** member
(`GpwDft` / `GPWEvaluator`): CP2K's `&PAIR_POTENTIAL R_CUTOFF` (doubled
convention, stored in Bohr; default = CP2K default 20 Å → 75.59 Bohr =
`kD3Cp2kPairCutoffBohr`). The pinned twin uses `R_CUTOFF 10.0` → 37.79 Bohr,
so the product path folds **exactly** the oracle's functional; energy, force
and stress all read the one member.

Sibling twins: `gpw-d3-atm-forces` (same geometry, ATM force pins — including
the CASE-3 image-triple and pbc-anchoring fixes this stress fold inherits),
`gpw-d3bj-stress` (two-body BJ stress pin, 12 Å cell — the pattern this case
follows), `gpw-d4-atm-forces` (D4 ATM forces; no CP2K D4 stress pin —
Arrhenius CP2K lacks libdftd4).

## Why the pin is a *difference* of two CP2K runs

DFT-D3 — with or without C9 — is geometry-only and never enters the
Kohn–Sham potential, so at **every strained cell** the converged density is
identical with `CALCULATE_C9_TERM` `.TRUE.` and `.FALSE.`, and the two-body
dispersion energy is identical too. CP2K's `STRESS_TENSOR NUMERICAL`
re-converges the SCF at each perturbed cell, hence

\[
\sigma^{\mathrm{CP2K}}(\mathrm{C9})-\sigma^{\mathrm{CP2K}}(\mathrm{2b})
 = -\Omega^{-1}\,\partial E_{\mathrm{ATM}}/\partial\varepsilon
 \Big|_{\mathrm{CP2K\ FD}},
\]

the pure three-body ATM stress: the (three-orders-larger) electronic stress,
the two-body dispersion stress, **and** the shared parts of CP2K's FD scheme
cancel in the difference. Total-energy difference of the two runs reproduces
\(E_{\mathrm{disp}}(\mathrm{C9})-E_{\mathrm{disp}}(\mathrm{2b})\) to
4.7e−15 Ha — the electronic cancellation is exact to solver noise.

## System

Verbatim the merged `gpw-d3-atm-forces` periodic twin:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **6 × 6 × 6 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | DFT-D3 zero damping, PBE: s6 = 1.0, s8 = 0.722, sr6 = 1.217, sr8 = 1.0; s9 = s6; ATM per run (`CALCULATE_C9_TERM` T/F) |
| Pair/ABC cutoff | `R_CUTOFF 10.0` Å → doubled 37.794 Bohr (`d3_rcut_bohr`; CP2K prints half: 18.90) |
| CP2K cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV cutoff | 150 Ry (both dispersion gates are geometry-only ⇒ cutoff-independent; the dual gate is grid-consistent by construction) |
| SCF | CP2K `EPS_SCF 1e-8` (tightened vs the 1e-6 forces twin — lowers the FD-difference noise floor); ATV `tol = 1e-6` |
| Stress step | ATV analytic/FD \(h=2\times10^{-3}\); in-regression ATM-only FD at **exactly CP2K's step** (`&NUMERICAL_STRESS` default `DX 0.001` Bohr ⇒ \(\varepsilon = DX/L = 8.8196\times10^{-5}\)) — see the step-sensitivity note below |

Strain at fixed fractional coordinates leaves the `accumulate_atm_c9` ANINT
wrap decisions invariant (positions and cell scale together), and the
truncated-boundary-set anchoring moves the ATM **energy** only at ~4e−19 Ha
(`gpw-d3-atm-forces` REFERENCE) ⇒ ≲3e−18 Ha/Bohr³ in the FD — far below
every gate here.

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-atm-stress/water_dimer_d3_atm_per6_stress.inp` (C9 T) and `water_dimer_d3_per6_stress.inp` (C9 F) |
| Output lines | final `Dispersion energy:` + `STRESS| Numerical stress tensor [bar]` table (20 dispersion evaluations per run: reference + 18 perturbed + final) |
| Wall | 54.5 s (C9 run), 18.0 s (two-body run) — CP2K internal timer |
| Date | 2026-07-31 |

### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (C9, per6) | −34.345965809817606 |
| Total (two-body, per6) | −34.345966594791001 |
| **E_disp (C9)** (printed) | **−0.00195809486938** |
| **E_disp (two-body)** (printed) | **−0.00195887984277** |

Both reproduce the `gpw-d3-atm-forces` periodic energy pins exactly (print
precision). ATM contribution +7.8497339e−7 Ha (repulsive — condensed-phase
sign); the total-energy difference reproduces it to 4.7e−15 Ha.

### Numerical stress tensors (`STRESS| Numerical stress tensor [bar]`)

| Component | C9 run | two-body run | Δ = C9 − 2b [bar] | Δ [Ha/Bohr³] |
| --------- | ------ | ------------ | ----------------- | ------------- |
| xx | 4.69756890548E+04 | 4.69807342693E+04 | −5.04521450 | **−1.7148335571E−08** |
| yy | 2.34423739140E+03 | 2.33148843324E+03 | +12.74895816 | **+4.3332828109E−08** |
| zz | 1.00716094815E+05 | 1.00713419919E+05 | +2.67489600 | **+9.0917867267E−09** |

Conversion: 1 Ha/Bohr³ = 2.9421015697×10⁸ bar (CODATA 2018). Off-diagonals
of both tables are ≤ 2.1e−5 bar (≈ 7e−14 Ha/Bohr³) — the numerical-stress
noise floor. Note the ATM stress is strongly anisotropic (yy largest) even
though the dimer axis is x: the compressed 6 Å cell's image triples dominate.

## Measured results (2026-07-31, this pin; `run_reg` target, OMP 8, loaded login node)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| E_disp (C9) vs CP2K (standalone) | 1e−9 Ha | **2.79e−10** |
| E_disp (two-body) vs CP2K | 1e−9 Ha | **2.83e−10** |
| E_disp (C9) product (`GPWEvaluator`) | 1e−9 Ha | **1.18e−10** |
| diag σ_ATM vs CP2K Δ | 5e−12 Ha/Bohr³ | **1.13e−13** (xx; yy 1.14e−14, zz 3.68e−14) |
| ATM engage floor max \|σ_ATM\| | ≥ 1e−9 Ha/Bohr³ | 4.33e−8 |
| dual gate \|σ_an − σ_fd\| (xx in e2e) | 2e−5 Ha/Bohr³ | **2.38e−7** |
| off-diagonal product | 1e−8 Ha/Bohr³ | ~0 (pass) |
| max \|σ_diag\| floor | > 1e−10 Ha/Bohr³ | 2.26e−4 |
| Wall (product + x-only FD twin) | ≤ 60 s | **40.8 s + 7.8 s** (9 SCF iters; load ≈ 22/48 cores) |

Product analytic diagonals (150 Ry, `tol = 1e−6`,
\(E_{\mathrm{tot}}\) = −34.301294491640 Ha):

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | +6.92003753×10⁻⁵ |
| \(\sigma_{yy}\) | −1.57391519×10⁻⁴ |
| \(\sigma_{zz}\) | +2.25590855×10⁻⁴ |

## The ATM strain-FD is step-locked (re-pin warning)

The two-body D3 sum passes a **smooth** fifth-order cutoff, so its strain-FD
is step-independent to ~1e−13 (`gpw-d3bj-stress` matched CP2K at a different
step). The ATM triple enumeration instead passes **hard** gates —
`r² < r_cut²` on each leg plus CP2K's empirical scaling gate
`(r²_AB r²_BC r²_CA) > 0.01 r_cut⁶` — so the included boundary set jumps by
~5e−11 Ha at discrete strains. An FD window that differs from the oracle's
straddles different gate crossings: at \(h = 10^{-4}\) the xx/zz components
were off by 1.6e−10 Ha/Bohr³ (staircase, not noise), while at exactly CP2K's
`DX 0.001` Bohr step the pin reproduces to 1.1e−13. Any re-pin must use the
oracle's `DX` (and note it if CP2K ever changes the default).

## Tolerance justification

- **E_disp 1e−9 Ha:** closed-form D3 + ATM sum with published parameters and
  the CP2K CN convention; the same cell measured 2.8e−10 in the
  `gpw-d3-atm-forces` pin. Only table transcription can move it. Also gates
  the periodic image-triple enumeration (CASE 1/2/3) under the stress input.
- **σ_ATM 5e−12 Ha/Bohr³:** the pin is a difference of two independently
  converged CP2K numerical-stress runs at `EPS_SCF 1e−8` sharing one FD
  scheme — electronic + two-body parts cancel to the print floor (~7e−14,
  the off-diagonal residuals). The residual at the shared `DX` step measured
  1.13e−13 (xx) against the 9.1e−9…4.3e−8 signals; 5e−12 sits ~40× above
  the measured residual and keeps ~1800× separation from the smallest
  signal, while absorbing plausible gate-crossing jitter of a re-pin on a
  different host (≲1e−12, the observed near-step spread).
- **Dual gate 2e−5 Ha/Bohr³:** same bound as `gpw-d3bj-stress` (measured
  7.4e−7 there, 2.38e−7 here): analytic fixed-P Lagrangian vs a fully
  re-converged SCF at `tol = 1e−6` — the residual is SCF-convergence noise,
  not the ATM fold.

## Dual gate

- **Correctness:** the pins above, plus the decisive analytic-vs-multi-SCF-FD
  identity on the same `d3 + d3_atm` method in this binary (x axis — the FD
  twin is gate overhead, not the product path), plus the cheap **periodic**
  H₂ unit twin `GpwStress.AnalyticMatchesFdWithD3Atm`
  (`tests/cpu/gpw/test_cpu_gpw_stress.cpp`) — lattice-image triples engage
  there (an isolated pair has no triple), and its short `d3_rcut_bohr`
  exercises the member through both stress paths' config copies.
- **Speed:** CP2K numerical-stress C9 twin 54.5 s (18 perturbed SCFs +
  reference, 2 MPI × 2 OMP, 300 Ry); Atomiverse product analytic pass
  **40.8 s** (one SCF + 7 fixed-P grid evaluations, OMP 8, 150 Ry, load ≈
  22/48 login node) → ratio ≈ **0.75**, inside the 10× fair wall. Settings
  caveat: cutoffs and core counts differ as stated — both produce the same
  deliverable (the ATM-folded cell stress of this system), which is the wall
  being compared.

## Notes

- The ATM term still contributes **no** `D3Result::virial` entry; the
  fixed-P stress path never consumes the virial — it rebuilds the dispersion
  energy at each strained cell, which is the fold this pin certifies. A
  future virial-path consumer (e.g. MD barostat coupling through the pair
  virial) would be its own queue row.
- `gpw-d4-atm-stress` is the D4 sibling (no CP2K pin possible — libdftd4
  absent; dftd4-CLI strained-cell energies policy, like `gpw-d4-atm-forces`).
- The periodic pin deliberately keeps `R_CUTOFF 10.0` from the forces twin:
  at the CP2K default (2 × 20 Å = 75.6 Bohr) the 6 Å-cell triple enumeration
  is ~40× more images for the same code coverage; both members of the pair
  and the ATV product use the same doubled-cutoff convention via
  `d3_rcut_bohr`.
