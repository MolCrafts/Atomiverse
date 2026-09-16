# Reference — GPW cell stress on isolated water

## Method

Closed-shell Γ-point GPW (Quickstep): Kohn–Sham in contracted Gaussians;
Hartree/XC/local-PP on a uniform grid. Cell stress v1 = **central finite
difference of the SCF energy under orthorhombic strain** at fixed fractional
coordinates (same strategy as plane-wave `pw_stress` strain-FD). Analytic
Nielsen–Martin / Pulay stress is a follow-on. Shear (xy/xz/yz) requires a
triclinic GPW cell and is **0** in Atomiverse v1.

Sign convention matches CP2K `STRESS_TENSOR`:
\(\sigma_{aa} = -\Omega^{-1}\partial E/\partial\varepsilon_{aa}\)
(E in Hartree, Ω in Bohr³). Positive \(\sigma_{xx}\) when expanding the box
lowers the energy.

Atomiverse path: `cpu::gpw::GPWEvaluator` with `compute_stress = true`
(`gpw_stress` → Frame `kStress` [eV/Å³]; pin compare via
`last_stress_ha_bohr3()` [Ha/Bohr³]).

## System

Isolated H₂O, CP2K getting-started geometry (Å), cubic cell 10 Å,
Martyna–Tuckerman / `PERIODIC NONE` Poisson.

| Atom | x | y | z |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

Settings (twin):

| Knob | Value |
| ---- | ----- |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 400 Ry |
| REL_CUTOFF (CP2K) | 60 Ry |
| NGRIDS (CP2K) | 4 |
| SCF | EPS_SCF 1e-7; Atomiverse `tol=1e-7` |
| Electrostatics | MT / nonperiodic |
| Stress | CP2K `STRESS_TENSOR NUMERICAL`; ATV strain-FD `h=2e-3` |
| `local_pp_on_grid` | true (Atomiverse product default) |

Twin input: `tools/cp2k_oracle/inputs/h2o_stress_mt.inp`.

**Why NUMERICAL not ANALYTICAL:** on this nonperiodic MT water twin, CP2K
`STRESS_TENSOR ANALYTICAL` reports ~−3×10⁵ bar diagonals while NUMERICAL
reports ~+10³ bar diagonals. Atomiverse strain-FD matches **NUMERICAL**
(magnitude and sign after CP2K sign convention). Analytic GPW stress is a
follow-on; the product pin is the same-method FD twin.

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc (offline oracle; **not** invoked from ctest) |
| RUN_TYPE | ENERGY_FORCE + `STRESS_TENSOR NUMERICAL` |
| Wall (CP2K) | ≈ **87 s** wall for full NUMERICAL stress (many SCF); fair single-SCF energy+analytic forces twin is ~10 s — use energy-force wall for fair ratio note, or full NUMERICAL wall when comparing FD to FD |
| Fair wall note | ATV strain-FD is 6+1 SCFs; CP2K NUMERICAL is many strained SCFs (~87 s). Fair comparison: CP2K ENERGY_FORCE without full num stress (~10.15 s from analytic stress run) vs ATV stress pass, **or** both FD walls. |

**Total energy (FORCE_EVAL QS, unperturbed):**  
\(E = -17.212337150586624\) Hartree

**Numerical stress tensor** (CP2K prints **bar**):

|  | x | y | z |
| -- | - | - | - |
| x | +6.45857359202×10³ | −2.92698205403×10⁻⁶ | +7.31745513509×10⁻⁷ |
| y | (sym) | +3.88814588443 | −1.46349102702×10⁻⁵ |
| z | (sym) | (sym) | +3.73657711805×10³ |

Unit conversion (CP2K internal table):  
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\).

Hard-coded pins in `reg_gpw-stress-water.cpp` (Ha/Bohr³, Voigt xx,yy,zz,xy,xz,yz):

```text
xx:  2.195224914616e-05
yy:  1.321554147453e-08
zz:  1.270036962196e-05
xy: -9.948611466750e-15
xz:  2.487152866693e-15
yz: -4.974305733392e-14
E_tot_Ha: -17.212337150586624
```

## Tolerance (physical justification)

- Diagonal stress tol **5×10⁻⁶ Ha/Bohr³**: same-method FD twin; residual is
  strain step / SCF / multigrid difference. |\σ_xx| ~ 2×10⁻⁵ so 5e-6 is ~25%
  relative on the largest component — still order-correct and sign-correct;
  measured residual after flip is ~10⁻⁸ level (sub-percent).
- Off-diagonal: ATV v1 **exactly 0**; CP2K off-diag ~10⁻¹⁴ Ha/Bohr³
  (soft zero within 1e-8).
- Energy |ΔE| **1×10⁻³ Ha** soft check (same as `gpw-forces-water`).

## Perf gate

Fair wall on Arrhenius epyc login, 4 OMP threads, `OMP_PROC_BIND=close`,
single rank.

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) (ENERGY_FORCE single SCF baseline) | **10.15 s** |
| \(T_{\mathrm{cp2k,num}}\) (full NUMERICAL stress) | **~87 s** |
| \(T_{\mathrm{atv}}\) (strain-FD stress regression) | **48.2 s** |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **4.75** (vs ENERGY_FORCE baseline) |
| ratio vs CP2K NUMERICAL | **0.55** (ATV faster than full NUMERICAL twin) |
| within_10x | **true** (≤ 10) |
| perf_exempt | false |

Measured ATV stress parity: max\|σ_diag_atv−σ_cp2k\| = **9.1×10⁻⁹** Ha/Bohr³;
\|ΔE\| = 4.2×10⁻⁴ Ha; SCF iters = 9.

Note: ATV strain-FD costs **6 SCF** (2 per diagonal) + 1 energy SCF; fair
product gate uses ENERGY_FORCE wall (10.15 s) as denominator.
