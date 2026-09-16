# Reference — periodic water-dimer GPW PBE-D4+ATM analytic cell stress

## Method

Γ-point **GPW** (PBE) with additive **DFT-D4** including the
Axilrod–Teller–Muto **three-body** term (\(s_9 = 1\), dftd4 `pbe-D4-ATM`) on
a periodic water dimer — the stress companion of `gpw-d4-atm-forces` and the
ATM extension of `gpw-d4-stress`.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path
(`gpw_stress`): one SCF → \(P, W\); central FD of the non-self-consistent
Lagrangian under orthorhombic diagonal strain at fixed fractional
coordinates, with geometry-only
\(E_{\mathrm{disp}}^{\mathrm{D4+ATM}}(\{\mathbf{R}\},\mathbf{h})\)
(**s9 = 1** when `d4_atm`) rebuilt at each strained cell;
\(\sigma_{aa} = -\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
(CP2K `STRESS_TENSOR` sign).

Public path: `cpu::gpw::GPWEvaluator` with `d4 = true`, `d4_atm = true`,
`compute_stress = true` → Frame `kStress` [eV/Å³];
`last_stress_ha_bohr3()` [Ha/Bohr³].

## System

Same geometry as `gpw-d4-atm` / `gpw-d4-atm-forces` / `gpw-d4-stress`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms** (order **O H H O H H**) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); **ATM on**: \(s_9=1\), alp = 16, disp3 cutoff 40 Bohr |
| Cutoff | 200 Ry |
| Stress step | \(h = 2\times10^{-3}\) |

## Offline pins (dftd4 3.7.0)

### Hard + decisive: strained-cell dispersion energies

The ATM stress contribution is **~10⁻¹⁰ Ha/Bohr³** on this system — three
orders below the GPW analytic-vs-FD dual-gate residual (~7×10⁻⁷), so only
dedicated energy pins at the exact strain samples resolve it. For each axis,
cell row a **and** the Cartesian a-components scaled by \(1 \pm h\) (the
`scale_axis` fixed-fractional convention), dftd4 3.7.0 printed energies
(`tools/cp2k_oracle/inputs/gpw-d4-atm/pin_strain_*_{atm,2b}.json`):

| Strain | s9=1 [Ha] | two-body (`--mbdscale 0`) [Ha] |
| ------ | --------- | ------------------------------ |
| x, +h | −9.9375767649976510e−04 | −9.9281871131614977e−04 |
| x, −h | −9.9733212469993015e−04 | −9.9635829706912161e−04 |
| y, +h | −9.9553758220206315e−04 | −9.9456763760422703e−04 |
| y, −h | −9.9555200940363935e−04 | −9.9460945924979903e−04 |
| z, +h | −9.9558989265599194e−04 | −9.9463925734105045e−04 |
| z, −h | −9.9549659214095867e−04 | −9.9453454831018983e−04 |

Base-cell s9=1 pin (product `e_disp` gate): **−9.9554475834240290e−04 Ha**
(`pin_periodic_atm.json`, shared with `gpw-d4-atm`).

Derived oracle dispersion stress, σ_aa = −[E(+h) − E(−h)]/(2hΩ),
Ω = 11661.119488 Bohr³ — the ATM-only part is the physics this case gates:

| Axis | σ^disp(s9=1) [Ha/Bohr³] | σ^ATM = σ(s9=1) − σ(2b) |
| ---- | ----------------------- | ----------------------- |
| xx | −7.6631754863e−08 | **−7.474078e−10** |
| yy | −3.0930138378e−10 | **+5.873030e−10** |
| zz | +2.0002478134e−09 | **−2.445845e−10** |

All gates are on the oracle's **printed energies** — never its gradient
(dftd4 3.7.0's periodic ATM gradient is non-conservative by ~2.4×10⁻⁸
against its own energy; documented in `gpw-d4-atm-forces/REFERENCE.md`).

### Why not CP2K QS?

**CP2K 2026.1** on Arrhenius is built **without** DFTD4 (`TYPE DFTD4`
aborts). The hard pin is **libdftd4 / dftd4 3.7.0**, the same backend CP2K
uses when DFTD4 is linked — same policy as `gpw-d4-atm` /
`gpw-d4-atm-forces` / `gpw-d4-stress`.

### Re-run (offline only — never ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
# For each axis a and sign ±: scale lattice row a AND Cartesian a-components
# of tools/cp2k_oracle/inputs/gpw-d4-atm/POSCAR by (1 ± 2e-3), save as
# POSCAR in its own directory (dftd4 detects format by filename), then:
dftd4 -f pbe --json pin_strain_<a>_<pm>_atm.json POSCAR
dftd4 -f pbe --mbdscale 0.0 --json pin_strain_<a>_<pm>_2b.json POSCAR
```

## Tolerances

| Check | Tol |
| ----- | --- |
| strained \|E_disp − pin\| (s9=1 and 2b, 12 values) | **1×10⁻⁹** Ha |
| \|σ^disp(s9=1) − FD(oracle E)\| and \|σ^ATM − Δpin\| per axis | **1×10⁻¹²** Ha/Bohr³ |
| ATM non-vacuity max\|σ^ATM_pin\| | > **5×10⁻¹⁰** Ha/Bohr³ (pin 7.47×10⁻¹⁰) |
| product \|e_disp − ATM base pin\| | **1×10⁻⁸** Ha |
| dual gate \|σ_xx^an − σ_xx^fd\| (multi-SCF strain-FD twin) | **2×10⁻⁵** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| max\|σ_diag\| floor | > **1×10⁻¹⁰** Ha/Bohr³ |
| wiring \|(σ_xx(ATM) − σ_xx(2b)) − σ^ATM_xx\| | **3×10⁻¹⁰** Ha/Bohr³ (noise 7×10⁻¹¹ from independently converged reference densities at SCF tol 10⁻⁶; dropping the s9 fold leaves 7.5×10⁻¹⁰ → FAIL) |

## Atomiverse numbers (first green, 2026-07-31)

| Quantity | Value |
| --- | --- |
| strained-energy parity max\|d pin\| (12 values) | **6.30×10⁻¹¹ Ha** (Bohr-conversion constant) |
| σ^disp(s9=1) parity max | 5.19×10⁻¹⁵ Ha/Bohr³ |
| σ^ATM parity max | **3.65×10⁻¹⁶ Ha/Bohr³** |
| E (Γ PBE-D4+ATM 200 Ry) | −34.331326946882 Ha |
| e_disp product \|d pin\| | 2.76×10⁻¹¹ Ha |
| σ_xx analytic (d4+ATM) | −2.17593330×10⁻⁵ Ha/Bohr³ |
| σ_yy analytic | +2.00729249×10⁻⁵ Ha/Bohr³ |
| σ_zz analytic | +5.82666928×10⁻⁵ Ha/Bohr³ |
| dual gate \|σ_xx^an − σ_xx^fd\| | **7.392×10⁻⁷** Ha/Bohr³ (same class as `gpw-d4-stress` 7.39×10⁻⁷) |
| wiring \|Δσ_xx − σ^ATM_xx\| | 7.05×10⁻¹¹ Ha/Bohr³ |
| walls (OMP=8, Release) | product 24.4 s; FD twin (x, warm) 4.9 s; 2b analytic (x, warm) 6.1 s |

Unit twin: `tests/cpu/gpw/test_cpu_gpw_dftd4.cpp`
`WaterDimerPeriodicStrainedAtmMatchesDftd4` (strained x+h s9=1 pin at 1×10⁻⁹;
measured 3.8×10⁻¹²).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | Arrhenius CP2K lacks DFTD4 — no fair QS `STRESS_TENSOR` wall exists. Hard gate is dftd4 printed-energy parity at the strain samples (same policy as `gpw-d4-atm` / `gpw-d4-atm-forces` / `gpw-d4-stress`). Case wall ~36 s is dominated by the one product SCF + FD twin, not by the ATM term. |

## What is checked

1. ATV `dftd4_energy(s9=1)` and `(s9=0)` at all 6 strained cells match the
   dftd4 3.7.0 printed energies (1×10⁻⁹ Ha).
2. The dispersion cell stress and its **ATM-only contribution** from those
   energies match the FD over the oracle's printed energies (1×10⁻¹²
   Ha/Bohr³), and the ATM contribution is non-vacuous (>5×10⁻¹⁰ pin).
3. Product GPW `d4 + d4_atm` analytic fixed-P stress: converges, `e_disp`
   matches the ATM base pin, off-diag ≈ 0, `kStress` unit conversion.
4. Dual gate: analytic vs multi-SCF strain-FD twin on the same d4+ATM
   method (σ_xx, warm-started).
5. Wiring: σ_xx(d4_atm) − σ_xx(d4-only) equals the standalone σ^ATM_xx —
   the `s9` fold inside the strained-cell dispersion rebuild of
   `gpw_stress` is exercised, not just the energy path.
