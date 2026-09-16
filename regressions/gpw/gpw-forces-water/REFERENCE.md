# Reference — GPW analytic forces on isolated water

## Method

Closed-shell Γ-point GPW (Quickstep): Kohn–Sham in contracted Gaussians;
Hartree/XC/local-PP on a uniform grid. Ionic forces = Hellmann–Feynman +
Pulay (energy-weighted density matrix · dS) + grid / core terms. See
VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter, *Comput. Phys.
Commun.* **167**, 103 (2005).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `compute_forces = true`
(`gpw_forces_analytic` one SCF + gradient; `local_pp_on_grid = true` default).

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
| `local_pp_on_grid` | true (Atomiverse product default) |

Twin input: `tools/cp2k_oracle/inputs/h2o_forces_mt.inp`.

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc (offline oracle; **not** invoked from ctest) |
| RUN_TYPE | ENERGY_FORCE |
| Wall (CP2K) | ≈ **11.2 s** wall (`mpirun -n 1`, `OMP_NUM_THREADS=4`) |

**Total energy (FORCE_EVAL QS):**  
\(E = -17.21233715058673\) Hartree

**Atomic forces [Hartree/Bohr]** (CP2K `FORCES| Atomic forces`):

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O | \(-6.62274644\times 10^{-8}\) | \(-1.03414433\times 10^{-6}\) | \(-8.96660013\times 10^{-2}\) |
| H1 | \(+5.16349669\times 10^{-2}\) | \(-2.88080683\times 10^{-9}\) | \(+4.49073908\times 10^{-2}\) |
| H2 | \(-5.16349704\times 10^{-2}\) | \(-2.33864544\times 10^{-10}\) | \(+4.49073720\times 10^{-2}\) |

Sum \(|\sum F|\) residual printed by CP2K: \(\sim 1.5\times 10^{-4}\) Ha/Bohr
(grid / SCF residual; not a physical net force on a free molecule at
numerical zero).

Hard-coded pins in `reg_gpw-forces-water.cpp` (Ha/Bohr):

```text
O:  -6.62274644e-08, -1.03414433e-06, -8.96660013e-02
H1:  5.16349669e-02, -2.88080683e-09,  4.49073908e-02
H2: -5.16349704e-02, -2.33864544e-10,  4.49073720e-02
E_tot_Ha: -17.21233715058673
```

## Tolerance (physical justification)

- Component force tol **1×10⁻³ Ha/Bohr**: same-method twin (Quickstep GPW);
  residual is dominated by grid / multigrid / mixing differences, not chemical
  accuracy. CP2K and Atomiverse both use product MT + PBE + MOLOPT/GTH; 1e-3
  is well below physical |F| (~0.09 Ha/Bohr on O_z).
- Energy |ΔE| **1×10⁻³ Ha** soft check against CP2K FORCE_EVAL total
  (−17.21233715 Ha); product multigrid residual is ~4×10⁻⁴ Ha on this twin.
- Translational residual: \(\|\sum_A\mathbf{F}_A\| \le 5\times 10^{-4}\)
  Ha/Bohr (CP2K print residual ~1.5×10⁻⁴ on the same twin; not a physical
  free-molecule zero at machine precision).

## Perf gate

Fair wall on Arrhenius epyc login (same host class as pin), 4 OMP threads,
`OMP_PROC_BIND=close`, single rank. ATV timed as `./build/regressions/reg_gpw-forces-water`
(one Evaluate pass with analytic forces). CP2K timed as
`mpirun -n 1 cp2k.psmp -i h2o_forces_mt.inp` (ENERGY_FORCE).

| Metric | Value |
| ------ | ----- |
| \(T_{\mathrm{cp2k}}\) | **11.2 s** wall |
| \(T_{\mathrm{atv}}\) | **39.9 s** wall |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | **3.56** |
| within_10x | **true** (≤ 10) |
| perf_exempt | false |

Measured ATV force parity: max\|F_atv−F_cp2k\| = **3.42×10⁻⁴** Ha/Bohr;
\|∑F\| = 2.1×10⁻⁴ Ha/Bohr; \|ΔE\| = 4.2×10⁻⁴ Ha; SCF iters = 9.
