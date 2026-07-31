---
slug: gpw-d3-atm-stress
status: approved
created: 2026-07-31
revised: 2026-07-31
domain: gpw / dispersion stress
queue: cp2k-method-loop
deferred: false
depends_on: [gpw-d3-atm-forces]
unblocks: [gpw-d3-atm-stress]
grilled: false
---

# Spec: `gpw-d3-atm-stress` — analytic fixed-P cell stress with DFT-D3 ATM (C9) folded

**Slug:** `gpw-d3-atm-stress`
**Domain:** GPW dispersion cell stress (CPU)
**Queue:** cp2k-method-loop, wave 2 (hold "behind ATM force pins" cleared —
`gpw-d3-atm-forces` merged with the CASE-3 image-triple fix + pbc anchoring)
**Oracle:** CP2K 2026.1 (`STRESS_TENSOR NUMERICAL`, offline via
`tools/cp2k_oracle`; extends the `gpw-d3-atm-forces` per-6 Å input)

## Summary

The GPW fixed-P (Hellmann–Feynman) cell-stress path already **rebuilds the
geometry-only DFT-D3 energy at each strained cell with `cfg.d3_atm`
propagated** (`src/cpu/gpw/internal/gpw_stress.cpp`: both `energy_fixed_P`
and `energy_fixed_P_k` call `dftd3_pbe(..., cfg.d3_atm)`; `copy_gpw` copies
`d3_atm` so the multi-SCF `gpw_stress_fd` oracle folds it too). What is
missing is **evidence**: no CP2K pin isolates the three-body
Axilrod–Teller–Muto contribution to the stress, and no unit twin covers
`d3_atm` under strain (the shipped stress twins are D4 / D3(BJ) two-body /
MBD). The `gpw-d3-atm-forces` REFERENCE explicitly deferred the ATM
**stress** pin to this row.

Deliverable = dedicated pin + regression + unit twin, same pattern as the
shipped `gpw-d3bj-stress`:

1. **Offline CP2K pin (delta construction).** Two CP2K 2026.1 runs on the
   periodic 6 Å water-dimer cell of `gpw-d3-atm-forces`
   (`PERIODIC XYZ`, `R_CUTOFF 10.0`, PBE, D3 zero damping), identical except
   `CALCULATE_C9_TERM` `.TRUE.` / `.FALSE.`, both with
   `STRESS_TENSOR NUMERICAL` and `EPS_SCF 1e-8`. D3 is geometry-only, so the
   SCF density is identical in both runs at every perturbed cell: the
   electronic **and two-body dispersion** stress cancel exactly in
   σ(C9) − σ(2b), leaving the pure ATM stress
   −Ω⁻¹ ∂E_ATM/∂ε (CP2K's own FD scheme, which cancels too).
2. **ATV parity gate.** Central strain-FD of the standalone
   `dftd3_pbe(..., calculate_c9)` energy **difference** (C9 on − off) at
   fixed fractional coordinates vs the CP2K delta — the identical fold the
   fixed-P stress path applies. Also pin E_disp (ATM on and off) against the
   printed `Dispersion energy` of both runs (re-checks the periodic
   image-triple enumeration under the stress inputs).
3. **Product path.** `GPWEvaluator{d3 = true, d3_atm = true,
   compute_stress = true, stress_use_fd = false}` on the same cell:
   analytic fixed-P stress finite, off-diagonals ~0 (orthorhombic), `kStress`
   eV/Å³ conversion checked.
4. **Dual gate.** Product analytic fixed-P stress vs multi-SCF strain-FD
   (`gpw_stress_fd`) on the same `d3 + d3_atm` method (x axis in the e2e to
   hold the 60 s budget), plus a cheap **unit twin**
   `GpwStress.AnalyticMatchesFdWithD3Atm` in
   `tests/cpu/gpw/test_cpu_gpw_stress.cpp` — a small **periodic** cell so
   lattice-image triples (CASE 2/3) actually engage (isolated H₂ has no
   triple).

## Why the pin is a difference of two CP2K runs

Same argument as `gpw-d3bj-stress` / `gpw-d3-atm-forces`: D3 (any damping,
with or without C9) never enters the Kohn–Sham potential. Both members of the
pair share one input except the C9 flag, one parallelization, one FD scheme —
everything except E_ATM cancels beyond solver noise. This isolates exactly
the quantity this row adds evidence for: the ATM contribution to the cell
stress.

## Geometry (from the merged `gpw-d3-atm-forces` periodic pin)

Water dimer, 6 atoms O H H O H H; O (0,0,0); H (0.758602, 0, ±0.504284);
O (3,0,0); H (3.758602, 0, ±0.504284) Å; cell 6 × 6 × 6 Å `PERIODIC XYZ`;
DZVP-MOLOPT-SR-GTH; GTH-PBE-q6/q1; PBE; CUTOFF 300 (CP2K) / 200 Ry (ATV —
dispersion gates are geometry-only, cutoff-independent; the dual gate is
grid-consistent by construction); D3 zero damping s6=1, s8=0.722, sr6=1.217,
s9=s6; `R_CUTOFF 10.0` (doubled convention → 37.79 Bohr) so the 6 Å-cell
triple enumeration stays cheap.

Strain FD at fixed fractional coordinates leaves the ANINT wrap decisions of
`accumulate_atm_c9` invariant (positions and cell scale together), and the
truncated-boundary-set anchoring moves the ATM **energy** only at the ~4e−19
Ha level (`gpw-d3-atm-forces` REFERENCE), i.e. ≲3e−18 Ha/Bohr³ in the FD —
far below every gate here.

## Expected signal / tolerances (confirm at pin time)

- E_ATM(per6) = +7.85e−7 Ha (repulsive) from the forces pin; the ATM stress
  delta is expected O(1e−9) Ha/Bohr³ diagonal (r⁻⁹ scaling → ∂E/∂ε ≈ few×E,
  Ω ≈ 1458 Bohr³).
- delta-sigma parity gate target ~1e−12 Ha/Bohr³ (queue says ~1e−13;
  set from the measured residual with ≥30× separation from the signal —
  the d3bj-stress twin measured 3.9e−14 against 1.15e−7 with tol 1e−10).
- E_disp pins: 1e−9 Ha (matches the forces-pin class; measured 2.8e−10
  there on the same cell).
- Dual gate |σ_an − σ_fd|: ≤ ~1e−6 … 2e−5 Ha/Bohr³ (gpw-d3bj-stress gate
  scale; the residual is SCF-convergence noise at tol 1e−6, not the fold).
- Wall ≤ 60 s; T_atv/T_cp2k ≤ 10 (CP2K numerical stress on this cell is
  ~19 SCFs; the ATV product pass is one SCF + 7 fixed-P grid evaluations on
  a 6 Å cell — expected well inside).

## Out of scope

- Analytic ATM **virial** (`D3Result::virial` three-body term stays absent;
  the fixed-P stress path does not consume it — it rebuilds the energy at
  strained cells. A future virial-path consumer would be its own row).
- D4 ATM stress (sibling row), D3(BJ)+ATM combination pin (CP2K supports it,
  but the queue row extends the zero-damping ATM force input).
- k-point dispersion-stress pin (dispersion fold is k-independent;
  `energy_fixed_P_k` shares the same rebuild lines).

## Files

| Path | Action |
| ---- | ------ |
| `tools/cp2k_oracle/inputs/gpw-d3-atm-stress/water_dimer_d3_atm_per6_stress.inp` | new (C9 T, NUMERICAL stress) |
| `tools/cp2k_oracle/inputs/gpw-d3-atm-stress/water_dimer_d3_per6_stress.inp` | new (C9 F twin) |
| `regressions/gpw/gpw-d3-atm-stress/{REFERENCE.md,reg_gpw-d3-atm-stress.cpp}` | new |
| `regressions/CMakeLists.txt` | register `atv_add_regression(gpw gpw-d3-atm-stress THREADS 8)` |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | add `AnalyticMatchesFdWithD3Atm` periodic twin |
| `docs/developer/cp2k-method-queue.md` | row → done on merge |
