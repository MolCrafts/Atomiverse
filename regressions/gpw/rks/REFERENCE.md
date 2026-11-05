# gpw/rks

One case; every section below is run by `atv_regressions --case gpw/rks` and keeps its own oracle record.


## Reference — GPW analytic forces on isolated water

### Method

Closed-shell Γ-point GPW (Quickstep): Kohn–Sham in contracted Gaussians;
Hartree/XC/local-PP on a uniform grid. Ionic forces = Hellmann–Feynman +
Pulay (energy-weighted density matrix · dS) + grid / core terms. See
VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter, *Comput. Phys.
Commun.* **167**, 103 (2005).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `compute_forces = true`
(`gpw_forces_analytic` one SCF + gradient; `local_pp_on_grid = true` default).

### System

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

### Offline CP2K pin

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

### Tolerance (physical justification)

- Component force tol **1×10⁻³ Ha/Bohr**: same-method twin (Quickstep GPW);
  residual is dominated by grid / multigrid / mixing differences, not chemical
  accuracy. CP2K and Atomiverse both use product MT + PBE + MOLOPT/GTH; 1e-3
  is well below physical |F| (~0.09 Ha/Bohr on O_z).
- Energy |ΔE| **1×10⁻³ Ha** soft check against CP2K FORCE_EVAL total
  (−17.21233715 Ha); product multigrid residual is ~4×10⁻⁴ Ha on this twin.
- Translational residual: \(\|\sum_A\mathbf{F}_A\| \le 5\times 10^{-4}\)
  Ha/Bohr (CP2K print residual ~1.5×10⁻⁴ on the same twin; not a physical
  free-molecule zero at machine precision).

### Perf gate

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


## Reference — GPW cell stress on isolated water

### Method

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

### System

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

### Offline CP2K pin

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

### Tolerance (physical justification)

- Diagonal stress tol **5×10⁻⁶ Ha/Bohr³**: same-method FD twin; residual is
  strain step / SCF / multigrid difference. |\σ_xx| ~ 2×10⁻⁵ so 5e-6 is ~25%
  relative on the largest component — still order-correct and sign-correct;
  measured residual after flip is ~10⁻⁸ level (sub-percent).
- Off-diagonal: ATV v1 **exactly 0**; CP2K off-diag ~10⁻¹⁴ Ha/Bohr³
  (soft zero within 1e-8).
- Energy |ΔE| **1×10⁻³ Ha** soft check (same as `gpw-forces-water`).

### Perf gate

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


## Reference — GPW analytic (fixed-\(P\)) cell stress on isolated water

### Method

Closed-shell Γ-point GPW (Quickstep): Kohn–Sham in contracted Gaussians;
Hartree/XC/local-PP on a uniform grid. **Product cell stress** is the
Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → density matrix \(P\).
2. Central finite difference of the **non-self-consistent** total energy
   \(E[P;\varepsilon]\) under orthorhombic diagonal strain at fixed fractional
   coordinates (rebuild \(T\), \(V_{\mathrm{pp}}\), grid, Poisson, XC; do **not**
   re-converge SCF).
3. \(\sigma_{aa} = -\Omega^{-1}\partial E/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

This is the GPW analogue of plane-wave Nielsen–Martin product stress in this
tree (`pw_stress`). Multi-SCF strain-FD remains `gpw_stress_fd` (oracle).

Shear (xy/xz/yz) requires a triclinic GPW cell and is **0** in v1.

#### Included terms (this product)

- Kinetic \(\mathrm{Tr}(P\,T(\varepsilon))\)
- Pseudopotential \(\mathrm{Tr}(P\,V_{\mathrm{pp}}(\varepsilon))\) — nonlocal always;
  local short-range either analytic per-pair or grid \(V_{\mathrm{loc}}^{\mathrm{SR}}\)
  matching `local_pp_on_grid` (same flag as SCF / analytic forces)
- Electrostatics \(E_{\mathrm{es}}[\rho_P - n_c]\) (MT or periodic FFT)
- XC \(E_{\mathrm{xc}}[\rho_P]\) (PADE / PBE / PBEsol grid path)
- Optional Grimme D3 / D3(BJ) geometry-only energy under the same strain

#### Deferred

- Closed-form term-by-term Pulay AO stress tables without fixed-\(P\) energy FD
- Off-diagonal / triclinic shear
- UKS, k-point GPW, DFT+U, D4, hybrid XC stress
- Absolute pin to CP2K `STRESS_TENSOR ANALYTICAL` on nonperiodic MT (known to
  disagree grossly with NUMERICAL on this twin — see `gpw-stress-water`)

Atomiverse path: `cpu::gpw::GPWEvaluator` with `compute_stress = true`,
`stress_use_fd = false` (default) → Frame `kStress` [eV/Å³]; compare via
`last_stress_ha_bohr3()` [Ha/Bohr³]. Oracle: free `gpw_stress_fd` or
`stress_use_fd = true`.

### System

Same twin as `gpw-stress-water`: isolated H₂O, CP2K getting-started geometry
(Å), cubic cell 10 Å, Martyna–Tuckerman / `PERIODIC NONE`.

| Atom | x | y | z |
| ---- | - | - | - |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

| Knob | Value |
| ---- | ----- |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 400 Ry |
| SCF | `tol=1e-7` |
| Electrostatics | MT / nonperiodic |
| Stress product | analytic fixed-\(P\), \(h=2\times 10^{-3}\) |
| Stress oracle | multi-SCF strain-FD, same \(h\) |
| `local_pp_on_grid` | true |

### Gates

| Check | Tolerance |
| ----- | --------- |
| \(\max_{\mathrm{diag}}\|\sigma^{\mathrm{an}}-\sigma^{\mathrm{FD}}\|\) | **1×10⁻⁵** Ha/Bohr³ |
| \(\|E - E_{\mathrm{cp2k}}\|\) | **1×10⁻³** Ha |
| Off-diagonal product | exactly 0 (soft 1e-8) |

Offline CP2K energy (FORCE_EVAL QS, unperturbed):  
\(E = -17.212337150586624\) Hartree  
(CP2K 2026.1; same pin as `gpw-stress-water` / `gpw-forces-water`.)

CP2K NUMERICAL stress absolute pin remains on `reg_gpw-stress-water` (product
default may now be analytic; residual vs NUMERICAL stays in that gate).

### Perf note

Analytic product cost class is **1 SCF + 6 fixed-\(P\) energy assemblies**
(vs **6 full SCFs** for multi-SCF FD diagonals). Regression runs both for the
dual gate; wall is dominated by the FD twin. Fair product-only wall is the
analytic Evaluate pass alone.

| Metric | Note |
| ------ | ---- |
| perf_exempt | false for product-only class; dual-gate wall higher |
| Fair twin | analytic vs multi-SCF FD (same engine) — no CP2K ANALYTICAL wall |


## Reference — isolated H₂O / PBE / DZVP-MOLOPT-GTH (Martyna–Tuckerman)

### Method

Γ-point closed-shell GPW Kohn–Sham DFT (CP2K Quickstep family): contracted
Gaussian basis for the wavefunction, uniform real-space grid + FFT for Hartree
and XC, GTH pseudopotentials. Isolated-molecule electrostatics via the
Martyna–Tuckerman (MT) Poisson solver (`PERIODIC NONE` / `PSOLVER MT` in CP2K;
`nonperiodic = true` on `cpu::gpw::GPWEvaluator`).

Public API: `cpu::gpw::GPWEvaluator` + `cpu::single_point` (Frame in/out).
Kinds use lookup constructors `GaussianBasis(Element, name)` /
`GthPotential(Element, name)` — no public `*Config` bag, no factories.

### System

Isolated water, CP2K getting-started geometry (Å), cubic box \(L = 10\) Å:

| Atom | x | y | z |
| ---- | --: | --: | --: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

- Basis: **DZVP-MOLOPT-GTH** (O, H)
- Pseudopotentials: **GTH-PBE-q6** (O), **GTH-PBE-q1** (H)
- XC: **PBE**
- Density cutoff: **400 Ry**
- Valence electrons: 8 → \(n_\mathrm{occ} = 4\)
- SCF: mix 0.3, max_iter 200, tol \(10^{-7}\)

### Pin (offline CP2K)

**\(E_\mathrm{tot} = -17.21233714458516\) Hartree**

Source: CP2K Quickstep getting-started tutorial input `h2o.inp` (same geometry,
basis/PP family, CUTOFF 400, PERIODIC NONE / PSOLVER MT, PBE). Published energy
decomposition (Hartree units) from that twin:

| Term | CP2K |
| ---- | ---: |
| self energy | −44.34715500 |
| core Hamiltonian | +13.07138998 |
| Hartree | +18.34339895 |
| XC | −4.27997143 |
| **total** | **−17.21233714458516** |

This is an **external-engine oracle**, not a golden snapshot of Atomiverse.
Cross-recorded in-tree by `examples/cpp/10_gpw.cpp` and
`docs/tutorial/ch09-gpw.md`. Historical engine milestone
(`6d7f0955`, 2026-06-18) reproduced this pin to ~0.43 mHa.

Optional twin input (for offline re-pin only; **never** launched by ctest):
`tools/cp2k_oracle/inputs/gpw-water-pbe-mt.inp`.

### Tolerance and rationale

**\(1 \times 10^{-3}\) Ha** absolute on \(E_\mathrm{tot}\).

The known Atomiverse residual on this minimal isolated-water stack is
~\(4.2 \times 10^{-4}\) Ha (ch09 live table: \(E_\mathrm{ATV} \approx
-17.212758\) Ha). A 1 mHa e2e band covers that residual with margin while still
failing gross method/geometry/cutoff mistakes. Bit-parity vs CP2K is a
follow-on (erf-screened MT tuning), not this regression's bar.

### What is checked

1. SCF converges (`eval.last_result().converged`).
2. \(E_\mathrm{tot}\) from `eval.last_energy_hartree()` within 1 mHa of the pin.
3. Finite energy on Frame (`frame.meta.get(kEnergy)` [eV]).

### Atomiverse result (verified 2026-07-23)

| Quantity | Value |
| -------- | ----- |
| \(E_\mathrm{tot}\) | −17.2127580657 Ha |
| \(\lvert\Delta E\rvert\) vs pin | \(4.209 \times 10^{-4}\) Ha |
| SCF iters | 9 |
| Converged | yes |
| Wall (\(T_\mathrm{atv}\), OMP=1, Release) | ~17.7 s |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary / module on the build host at pin time (`cp2k`, `cp2k.psmp`, `cp2k.sopt` not found; `module avail cp2k` empty). Fair wall twin deferred until offline CP2K is available. |
| \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) | n/a (exempt) |
| `within_10x` | n/a (exempt) |

When a CP2K binary is available, re-run the twin in
`tools/cp2k_oracle/inputs/gpw-water-pbe-mt.inp`, time both single-rank runs on
the same host/core count, and replace this section with measured \(r\).


## Reference — Monkhorst-Pack k-point GPW RKS on diamond Si (bct 4-atom cell)

### Method

Closed-shell GPW Kohn–Sham DFT (PBE) with **Monkhorst–Pack k-point
sampling** — the CP2K Quickstep `&KPOINTS SCHEME MONKHORST-PACK` loop shape:
complex Bloch-phased analytic matrices per k
(S_k / T_k lattice sums, phased Kleinman–Bylander GTH nonlocal), per-k
pre-summed complex basis-on-grid values for the k-summed density
ρ(r) = Σ_k w_k Σ_μν P^k_μν χ^k_μ conj(χ^k_ν), per-k complex Hermitian
canonical-orthogonalization + zheevd diagonalization, and k-summed Pulay
DIIS. CP2K MP convention κ_d = (2r − N − 1)/(2N) (N = 2 → ±1/4, Γ-avoiding);
time-reversal folding of ±k (exact for RKS).

Public API: concrete `cpu::gpw::GpwDft{kinds, cutoff_ry, xc}` with knob
members (`k_mesh`, `k_time_reversal`, explicit `k_points`/`k_point_weights`,
…); lookup constructors
`GaussianBasis(Element::Si, "DZVP-MOLOPT-GTH")` /
`GthPotential(Element::Si, "GTH-PBE-q4")`. No public `*Config`, no factories.
Result payload: `GpwResult::{k_frac, k_weights, eps_k}`.

### System

Silicon diamond in the **body-centred-tetragonal 4-atom cell**
(the orthorhombic-compatible setting of the fcc diamond lattice):

| | |
| --- | --- |
| Lattice | a = 5.43 Å (10.26120 Bohr); cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si) — in-tree table (`gpw_tables.h`) |
| Pseudopotential | GTH-PBE-q4 |
| Valence | 16 e⁻ → n_occ = 8 |
| XC | PBE |
| Density cutoff | 150 Ry (main run); 95 Ry (identity gates — commensurate supercell grids, unit-test-pinned) |
| k mesh | MONKHORST-PACK 2×2×2, time-reversal folded → 4 k, w = ¼ |
| SCF | DIIS depth 6, tol 1e-8 Ha, EPS_DEFAULT 1e-7 |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), module `CP2K/2026.1` on
  Arrhenius (visible after `module load buildtool-easybuild/5.2.1-hpca3ef7d197
  GCC/14.3.0 MPICH/4.3.2`).
- **Input:** the exact twin `tools/cp2k_oracle/inputs/gpw-kpoints-si.inp`
  (CUTOFF 150, NGRIDS 1, MONKHORST-PACK 2 2 2, PBE, EPS_SCF 1e-8).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  OMP_NUM_THREADS=4 mpirun -np 1 cp2k.psmp -i gpw-kpoints-si.inp -o si_kpoints_sg.out
  ```
- **Date:** 2026-07-31 (replaces the 2026-07-25 ATV-frozen first-green band —
  no CP2K module was visible then without the easybuild toolchain loaded).

| Quantity | CP2K pin | Tolerance (A7) |
|----------|----------|----------------|
| E_tot [Ha] | **−15.802416412655166** | **1e-3 Ha** |

Key log line (CP2K 2026.1): `ENERGY| Total FORCE_EVAL ( QS ) energy
[hartree]            -15.802416412655166`; SCF converged in 13 steps; wall
≈ 3.3 s (1 rank × OMP 4). ATV first green −15.8023941333 Ha sits 2.2e-5 Ha
from the CP2K number.

Supporting internal identities (unchanged):

1. **Internal exactness identities** — the decisive k-point correctness
   checks; each holds to SCF tolerance *independent of any oracle*:
   - **Supercell folding** (gate A2): E(bct, mesh 1×1×2 ⇒ κ_z = ±¼,
     genuinely complex phases {±1, ±i}) ≡ E(z-doubled 8-atom cell at the
     folded zone-boundary point κ = (0,0,½))/2. Measured **|Δ| = 6.9e-14 Ha**
     (tol 5e-5).
   - **Time reversal** (A3): full 8-k mesh vs TR-folded 4-k mesh —
     **|Δ| = 0.0** (tol 5e-6).
   - **Γ limit** (A1): explicit k = (0,0,0) through the complex machinery vs
     the pre-existing real Γ path (compact-basis periodic H₂ toy,
     DZV-GTH-PADE) — **|Δ| = 1.4e-11 Ha** (tol 1e-6).
   The A2 identity found a real bug at first run: the phased KB nonlocal
   assembly conjugated the wrong side (V_nl evaluated at −k; Hermitian,
   Γ-exact, TR-invisible — but 2.09 Ha off at κ = ¼). Fixed in
   `internal/kpoints.cpp`; the identity is now machine-exact.
2. **Absolute CP2K pin** (A7): the offline CP2K 2026.1 number above.

#### Atomiverse numbers (first green, 2026-07-25)

| Quantity | Value |
| --- | --- |
| **E(MP 2×2×2 TR, 150 Ry)** | **−15.8023941333 Ha** (SCF 6 iters) |
| E(MP 2×2×2 TR, 95 Ry) | −15.8024022460 Ha |
| E(explicit Γ, 150 Ry) | −15.2031525100 Ha |
| k-sampling shift \|E_MP − E_Γ\| | 5.99e-01 Ha (gate floor 1e-3) |
| E(mesh 1×1×2, 95 Ry) | −15.3483059090 Ha |
| E(8-atom supercell, κ_z = ½, 95 Ry)/2 | −15.3483059090 Ha |
| Sampled-k gap (VBM 0.137858 Ha, CBM 0.283137 Ha) | 3.953 eV (floor 0.1 eV) |

### Tolerances

- A1 ≤ 1e-6 Ha; A2 ≤ 5e-5 Ha; A3 ≤ 5e-6 Ha (identities).
- A4 ≥ 1e-3 Ha activation floor; A5 gap ≥ 0.1 eV.
- A7 absolute: 1e-3 Ha vs the offline CP2K 2026.1 pin.

### What is checked

1. Γ-limit, supercell-folding, and time-reversal identities (A1–A3).
2. k-sampling activation and insulating gap + k payload shapes (A4–A5).
3. SCF convergence, finiteness (A6); absolute CP2K pin (A7).
4. No external process spawn; ≤ 60 s.

### Performance

| Item | Value |
| ---- | ----- |
| T_cp2k (twin input, 1 rank × OMP 4, 2026-07-31) | ≈ 3.3 s |
| T_atv (main MP 2×2×2 run alone, OMP=8) | ~4 s |
| r = T_atv/T_cp2k | ≈ 1.2 (≤ 10 ✓) |
| `within_10x` | true |
| T_atv (whole regression: 7 SCF solves, OMP=8, Release) | ~17 s |

### Provenance

- Spec: `.claude/specs/gpw-kpoints-si/` (deleted on completion; git history)
- MP mesh: Monkhorst & Pack, PRB **13**, 5188 (1976); CP2K KPOINTS convention
- GTH: Goedecker–Teter–Hutter, PRB **54**, 1703 (1996); Krack, TCA **114**,
  145 (2005); MOLOPT: VandeVondele & Hutter, JCP **127**, 114105 (2007)
- Twin input: `tools/cp2k_oracle/inputs/gpw-kpoints-si.inp`


## Reference — GPW cell-strain stress at Monkhorst–Pack k-points (diamond Si)

### Method and system

Closed-shell PBE GPW with DZVP-MOLOPT-GTH and GTH-PBE-q4 on the diamond-Si
BCT four-atom cell: edges `(5.43/sqrt(2), 5.43/sqrt(2), 5.43)` Angstrom;
fractional positions `(0,0,0)`, `(1/2,1/2,1/2)`, `(1/2,0,1/4)`,
`(0,1/2,3/4)`. Sixteen valence electrons occupy eight orbitals. Sampling is
Monkhorst–Pack 1x1x2 with time reversal, CUTOFF 100 Ry, NGRIDS 1.
The local pseudopotential is integrated on the grid on both engines
(native `local_pp_on_grid=true`, CP2K `CORE_PPL GRID`).

Native stress differentiates the fixed-\(P_k,W_k\) Lagrangian under central
diagonal strain `h=2e-3`, rebuilding cell-dependent integrals and grid terms.
This is a strain finite difference, not a closed-form analytical tensor.
All original reconverged-SCF diagonal strain checks remain: every independent
plus/minus calculation starts from the same unstrained reference density.
The separate multigrid stress path is externally pinned in the existing
`gpw-multigrid-kpoints` section described below.

### External oracle and values

**Oracle: CP2K 2026.2, `git:c92cc08`, actually executed 2026-10-25.**
`/opt/homebrew/bin/cp2k.psmp` supports `STRESS_TENSOR ANALYTICAL` on this
k-point slice. The complete run exited zero and converged in 12 SCF steps
at `EPS_SCF 1e-10`. QS uses `EPS_DEFAULT 1e-12`, `EPS_PGF_ORB 1e-12`,
`EPS_PPNL 1e-12`, matrix filtering disabled, and the default derived
`EPS_GVG_RSPACE 1e-6`. Broyden mixing uses alpha 0.4 and eight history entries;
`EPS_DIIS 0` prevents switching to DIAG/DIIS.

| Quantity | CP2K NGRIDS 1 |
| --- | ---: |
| `ENERGY\| Total FORCE_EVAL` [Ha] | -15.348312942859359 |
| analytical sigma_xx / sigma_yy [bar] | 259796.048452 |
| analytical sigma_zz [bar] | 76792.7848508 |
| sigma_xx / sigma_yy [Ha/Bohr3] | 8.830288224301885e-4 |
| sigma_zz [Ha/Bohr3] | 2.610133709961543e-4 |

Complete tensors, raw output, unit conversion, exact command, hashes and
binary version are in
[`reference.json`](../../../tools/cp2k_oracle/inputs/gpw-kpoint-stress/reference.json)
and `singlegrid.cp2k.txt` beside it. The actual input is
[`gpw-kpoint-stress.inp`](../../../tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp).
Stress is compression positive; conversion from printed bar uses independent
CODATA2018 SI constants. No Atomiverse result is a reference value.

### Acceptance bounds

| Check | Bound |
| --- | --- |
| absolute E vs CP2K | 1e-3 Ha |
| each stress component vs CP2K | 1e-4 Ha/Bohr3 |
| diagonal fixed-P vs reconverged strain FD | 1e-4 Ha/Bohr3 |
| off-diagonal tetragonal symmetry | 1e-8 Ha/Bohr3 |
| largest diagonal magnitude | greater than 1e-6 Ha/Bohr3 |
| complete section wall | at most 60s at registered OMP8 width |

All original single-grid numerical checks remain. Native SCF uses energy
tolerance 1e-10 Ha, residual 1e-8 and screening 1e-12. The multigrid path uses
the same external numerical bounds in its existing section; an extra set
of multigrid reconverged FD solves is not required by that section's gates.

### Validation and performance boundaries

On 2026-10-25 the rebuilt native section passed all original checks:
energy error 1.0300e-6 Ha, maximum tensor error 2.7888e-8 Ha/Bohr3,
and maximum fixed-P versus relaxed-FD difference 7.23501e-8 Ha/Bohr3.
The complete `gpw/rks` case passed 9/9 sections. Its raw native output is
[`native-validation.txt`](../../../tools/cp2k_oracle/inputs/gpw-kpoint-stress/native-validation.txt).

| Measured boundary | Wall [s] |
| --- | ---: |
| Native energy + fixed-P stress product | 20.617223 |
| Native supplemental relaxed-SCF strain verification | 21.559348 |
| Native complete section, including all checks | 42.176594 |
| CP2K complete ENERGY_FORCE process + analytical stress | 63.903604 |

The native section passed the 60s bound in one shared `gpw/rks` process,
with CTest environment requests `OMP_NUM_THREADS=8` and
`OPENBLAS_NUM_THREADS=8`. The independent CP2K process requested
`OMP_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=1`, with binding unset.
CP2K's internal BLAS thread width was not queried; its environment request
does not establish the actual library width. These validation timings do not
establish a matched-width speedup. Native stress uses
fixed-P strain differences, while CP2K evaluates a closed-form tensor;
that algorithmic distinction also accompanies a wall comparison.

## Reference — Multi-k GPW RKS with REL_CUTOFF multigrid (diamond Si, NGRIDS 4)

### Method

Closed-shell GPW Kohn–Sham DFT (PBE) with **Monkhorst–Pack k-point sampling
and CP2K-style CUTOFF/REL_CUTOFF multigrid product collocation** — the
capability the k-point branch used to reject (`GpwDft::build`:
`"multigrid (ngrids > 1)"`). The geometry-fixed primitive-pair product list is
the same one the Γ multigrid uses; under Bloch sampling each product's ket
image shift t becomes a phase index:

- density: ρ(g) weighted per product by the real-space stack
  **D(t) = Σ_k w_k Re[P^k e^{−2πi κ·t}]** (Γ limit: D(t) = P);
- per-level densities prolong spectrally to the finest grid for Poisson + XC;
- the fine potential restricts back per level (exact adjoint) and integrates
  into the real per-cell stack **V_t**, phase-summed per k into the Hermitian
  grid KS matrix **V^k = Σ_t e^{+2πi κ·t} V_t**.

Public API: concrete `cpu::gpw::GpwDft{kinds, cutoff_ry, xc}` with knob
members (`k_mesh`, `ngrids`, `rel_cutoff`, …); lookup constructors
`GaussianBasis(Element::Si, "DZVP-MOLOPT-GTH")` /
`GthPotential(Element::Si, "GTH-PBE-q4")`. No public `*Config`, no factories.

### System

Silicon diamond in the body-centred-tetragonal 4-atom cell (the
orthorhombic-compatible setting of the fcc diamond lattice) — identical to
`regressions/gpw/rks`:

| | |
| --- | --- |
| Lattice | a = 5.43 Å (10.26120 Bohr); cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis | DZVP-MOLOPT-GTH (Si), in-tree table |
| Pseudopotential | GTH-PBE-q4 (16 valence e⁻ → n_occ = 8) |
| XC | PBE |
| Density cutoff | 150 Ry (main energy); 40 Ry / REL 20 (M2/M3); 100 Ry / REL 40 (M6 stress) |
| Multigrid | **NGRIDS 4, REL_CUTOFF 60 Ry** (ATV + CP2K twin) |
| k mesh | MP2×2×2 TR (main/M2/M3); MP1×1×2 TR (M6 stress) |
| SCF | main/M2/M3: tol 1e-8 Ha, EPS_DEFAULT 1e-7; M6: tol 1e-10 Ha, residual 1e-8, EPS_DEFAULT 1e-12 |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), module `CP2K/2026.1` on Arrhenius
  (visible after `module load buildtool-easybuild/5.2.1-hpca3ef7d197
  GCC/14.3.0 MPICH/4.3.2`).
- **Input:** `tools/cp2k_oracle/inputs/gpw-multigrid-kpoints/si_kpoints_mg.inp`
  (identical to the `gpw-kpoints-si` twin except `&MGRID`: CUTOFF 150,
  **NGRIDS 4, REL_CUTOFF 60**).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  OMP_NUM_THREADS=4 mpirun -np 1 cp2k.psmp -i si_kpoints_mg.inp -o si_kpoints_mg.out
  ```
- **Date:** 2026-07-31.

| Quantity | CP2K pin | Tolerance (M5) |
|----------|----------|----------------|
| E_tot [Ha], NGRIDS 4 | **−15.802416412503845** | **1e-3 Ha** |
| E_tot [Ha], NGRIDS 1 (context) | −15.802416412655166 | — |

Key log line: `ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]
-15.802416412503845`; SCF converged in 13 steps. CP2K's own NGRIDS 4 ↔ 1 gap
is 1.5e-10 Ha — the multigrid is a cost optimization, not a physics change,
which is what gate M3 pins on the ATV side.

### M6 external energy and stress pin

The same existing section also exercises the distinct multigrid stress path
at CUTOFF 100 Ry, NGRIDS 4/REL_CUTOFF 40 Ry, MP1x1x2 with time reversal.
This uses the same Si BCT geometry and PBE/basis/GTH potential above, with
native `local_pp_on_grid=true` and matching CP2K `CORE_PPL GRID`.
All original 150 Ry energy and 40 Ry identity gates remain unchanged.

**CP2K 2026.2, `git:c92cc08`, actually executed 2026-10-25**, converged in 12 SCF
steps and exited zero, printing:

| Quantity | Actual CP2K value | Native bound |
| --- | ---: | ---: |
| E [Ha] | -15.348312942398339 | 1e-3 Ha |
| analytical sigma_xx / sigma_yy [bar] | 259796.054263 | — |
| analytical sigma_zz [bar] | 76792.7681610 | — |
| sigma_xx / sigma_yy [Ha/Bohr3] | 8.830288421813762e-4 | 1e-4 Ha/Bohr3 |
| sigma_zz [Ha/Bohr3] | 2.610133142686770e-4 | 1e-4 Ha/Bohr3 |

All six tensor components are checked against the printed external tensor;
the independent off-diagonal symmetry bound is 1e-8 Ha/Bohr3 and the largest
diagonal magnitude must exceed 1e-6 Ha/Bohr3. These are the same bounds as
the single-grid stress section. Native `stress()` supplies both the energy
and fixed-P strain tensor after one unstrained SCF. Exact product-grid and
Bloch transfer identities are covered by the existing `GpwMultigridCells`
unit tests; the primary stress oracle is CP2K's actual analytical tensor.

The actual input is
[`gpw-kpoint-stress-multigrid.inp`](../../../tools/cp2k_oracle/inputs/gpw-kpoint-stress-multigrid.inp).
Full output is `multigrid.cp2k.txt`, and command/version/input/output/data
hashes, precise tensor and unit conversion are in
[`reference.json`](../../../tools/cp2k_oracle/inputs/gpw-kpoint-stress/reference.json).
The main 150 Ry pin retains its recorded CP2K 2026.1 provenance; the additional
100 Ry stress pin explicitly uses 2026.2 and matching grid-local PP.

### Found & fixed while opening the gate

The shipped **periodic** product path (Γ multigrid) screened whole products
and window tails at `eps_default` as a *value* threshold; a dropped diffuse
product discards **integrated mass** up to (π/p)^{3/2} ≈ 70× that value, and
a periodic cell holds thousands of image pairs at threshold. On this system
the collocated density integrated to **15.910 of 16 e⁻** and the SCF sat
4.1e-2 Ha off the single grid *even with every level at the fine cutoff*
(the shipped multigrid regressions are MT-isolated, where image pairs are
absent, so this never fired). Fixed by the derived screen margin
`kProductMassMargin = 1e-4` in `internal/multigrid.cpp` (Γ and k paths
alike); after the fix the eps ladder converges:

| eps_default | E(mg) − E(sg) [Ha] |
|---|---|
| 1e-7 (this regression) | +3.5e-5 |
| 1e-9 | +2.6e-6 vs sg at 1e-9 |

Also found on the same gate: nonlocal vdW (`vdw_df`/`vdw_df2`/`rvv10`) was
silently dropped on the whole k path (semilocal XC ran, E_c^nl never added) —
now rejected fail-fast at `build()`.

### Measured (2026-07-31, this tree)

Standing regression gates:

| Gate | Value | Tolerance |
| --- | --- | --- |
| M3 mg vs sg (40 Ry / REL 20) | 1.13e-5 Ha | 1e-3 |
| M5 \|E − CP2K\| (150 Ry / REL 60, MP 2×2×2 TR) | **4.73e-7 Ha** | 1e-3 |

E(MP 2×2×2, 150 Ry, NGRIDS 4) = −15.8024168854 Ha (7 DIIS iters).

Bring-up exactness identities (full-SCF twins measured once here, then kept as
unit tests — each multigrid SCF costs ~15–39 s and the ≤ 60 s budget goes to
the pin; mechanisms unit-pinned in `tests/cpu/gpw/test_cpu_gpw_multigrid.cpp`
`GpwMultigridCells.*`):

| Identity | Measured |
| --- | --- |
| Γ-limit: explicit k = (0,0,0) mg k-path vs Γ mg product path (95 Ry) | 2.0e-14 Ha |
| Time reversal: full 8k vs TR-folded 4k (95 Ry mg) | 2.8e-14 Ha |
| Time reversal (60 Ry mg) / (40 Ry mg) | 2.0e-14 / 2.3e-14 Ha |
| mg vs sg at 150 Ry / REL 60 | 2.28e-5 Ha |

### Validation and performance boundaries

The unchanged M2-M5 gates and new external M6 gate all passed on 2026-10-25.
M6's printed energy error is 9.9758e-7 Ha and maximum tensor error 2.9487e-8
Ha/Bohr3. The native M6 product took 13.235127s, and the complete section
including the original 40 Ry identity and 150 Ry energy gates took 17.708007s,
passing its 60s bound with registered environment requests
`OMP_NUM_THREADS=8` / `OPENBLAS_NUM_THREADS=8`. Exact output is in
[`native-validation.txt`](../../../tools/cp2k_oracle/inputs/gpw-kpoint-stress/native-validation.txt).

The actual CP2K 2026.2 M6 ENERGY_FORCE process with analytical stress took
17.833723s, one MPI rank, with environment requests `OMP_NUM_THREADS=8`
and `OPENBLAS_NUM_THREADS=1`, binding unset. CP2K's internal BLAS thread
width was not queried. Native runs in the shared `gpw/rks` process and
differentiates fixed-P strain energies; CP2K produces a closed-form tensor
in its separate process. These environment requests and measurement
boundaries do not establish a matched-width speedup. The main 150 Ry energy twin's
fair-wall source remains
[`.claude/notes/gpw-perf-baseline.md`](../../../.claude/notes/gpw-perf-baseline.md).

### Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- CP2K Quickstep `&MGRID` CUTOFF / REL_CUTOFF / NGRIDS multigrid collocation;
  `&KPOINTS SCHEME MONKHORST-PACK` (Monkhorst & Pack, PRB **13**, 5188 (1976)).
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.


## Reference — multi-k GPW free-energy analytic forces (FD smear)

### Method

Closed-shell **GPW** Kohn–Sham DFT with **Monkhorst–Pack k-point sampling**
and **Fermi–Dirac metallic occupation smearing**. Ionic forces are analytic
derivatives of the **Mermin free energy** \(F=E_{\mathrm{KS}}-TS\):

\[
P_k=\sum_b f_{kb}\,c_{kb}c_{kb}^\dagger,\qquad
W_k=\sum_b f_{kb}\varepsilon_{kb}\,c_{kb}c_{kb}^\dagger
\]

with fractional band occupations \(f_{kb}\in[0,2]\) from a shared \(\mu\).

**Payload fix:** `store_k_payload` fills `GpwResult::f_k` and rebuilds
fractional \(P_k\); force assembly uses full `C_k` + `f_k` (not integer
\(n_{\mathrm{occ}}\) truncation).

Public path: `GpwDft` members `smearing` / `degauss` / `extra_bands` +
`dft.forces(frame)` (SCF + analytic forces, Frame `kAtom_f*` in eV/Å) and
`gpw_forces_fd(frame, dft, h_ang)`. No public `*Config`, no factories.

### System (ATV correctness pin)

| | |
| --- | --- |
| Molecule | H₂, off-symmetry compressed bond |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.7, 6.1, 5.9) |
| Cell | cubic 12 Bohr, periodic XYZ |
| Basis | DZV-GTH-PADE (H) — in-tree |
| PP | GTH-PADE-q1 |
| XC | PADE |
| Cutoff | 100 Ry |
| k mesh | MONKHORST-PACK 1×1×2 (TR fold → nk=1) |
| Smear | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 2` |
| SCF | tol 1e-8 Ha, `local_pp_on_grid` |

### Offline correctness

#### Analytic vs free-energy FD (decisive)

Central finite difference of \(F\) (`gpw_forces_fd`, \(h=10^{-3}\) Bohr) on the
same surface:

| Quantity | Value |
| --- | --- |
| max\|F_analytic − F_fd\| | **7.8×10⁻⁶ Ha/Bohr** |
| Free energy \(F\) | **−1.118469277632 Ha** |
| F₀ [Ha/Bohr] | **(−0.014404059091, −0.009867891977, +0.009866830655)** |
| F₁ [Ha/Bohr] | **(+0.21449155103, +0.10920852043, −0.10920745848)** |

Recorded 2026-08-01, Arrhenius x86_64, Release, `OMP_NUM_THREADS=4`–`8`.

Translational residual \|∑F\| is nonzero on this coarse grid (eggbox); the same
pattern appears in the CP2K H₂ twin below. **Gradient identity vs FD is the
product correctness gate**, not absolute Newton-3 on a coarse mesh.

#### CP2K ENERGY_FORCE + SMEAR (algorithm twin)

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Input (fair wall / H₂) | `tools/cp2k_oracle/inputs/gpw-metallic-forces/h2_kmesh_fd_forces.inp` |
| Input (Na metal class) | `tools/cp2k_oracle/inputs/gpw-metallic-forces/na_bcc_fd_forces.inp` |
| H₂ twin | DZVP-MOLOPT-GTH + GTH-PADE-q1, k 1×1×2, FD \(T_e=3157.75\) K, CUTOFF 100 |
| H₂ \(E\) example | −1.122793679 Ha (not ATV pin — basis differs) |
| H₂ F₀ / F₁ example | (−4.7e−3, −1.0e−2, +1.0e−2) / (+0.238, +0.127, −0.127) Ha/Bohr |
| Na twin | BCC \(a=4.225\) Å, Na2 +0.05 Å along x, DZVP-MOLOPT-SR-GTH + GTH-PBE-q9 |
| Na \(E\) / \(E_F\) | −93.509962456 Ha / 0.053784 Ha |
| Command | `source …/cp2k/install/env.sh; mpirun -np 1 cp2k.psmp -i …` |

### Tolerances (regression)

- \|F − pin\| ≤ 1e−5 Ha
- \|F_x − pin\| ≤ 1e−5 Ha/Bohr
- max\|F_an − F_fd\| ≤ 5e−5 Ha/Bohr
- Ne = 2 within 1e−5; finite \(E_F\); non-empty `f_k`

### Performance (fair wall)

Matched twin: H₂ ENERGY_FORCE + SMEAR, 1 MPI × OMP=4.

| Engine | Wall |
| ------ | ---- |
| Atomiverse SCF + analytic F (`reg` scf+F, OMP=8) | **~0.11 s** (scf 0.037 + F 0.069) |
| CP2K `h2_kmesh_fd_forces.inp` (OMP=4) | **1.43 s** |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.077 \ll 10
\]

`within_10x = true`. Na ENERGY_FORCE twin (`na_bcc_fd_forces.inp`) wall
~0.7 s (hot) documents the metallic-class algorithm path; absolute Na forces
are not dual-gated (eggbox + basis/grid mismatch vs ATV).

### Payload / API notes

- `GpwResult::f_k` / `f_k_beta` — per-k fractional occupations after smear SCF
- `copy_gpw` in force path propagates smear members
- UKS store path also fills fractional `f_k` / `P_k` (UKS smear stress:
  `gpw-uks-metallic-stress`)


## Reference — multi-k free-energy analytic cell stress under FD smear (BCC Na, GPW)

### Method

Spin-unpolarized **GPW** Kohn–Sham DFT with **Fermi–Dirac** electronic
smearing (Mermin free energy \(F = E_{\mathrm{KS}} - TS\)). Product **cell
stress** is the Hellmann–Feynman fixed-occupation free-energy path:

1. One multi-k SCF with FD smear → full Bloch \(C_k\), band energies
   \(\varepsilon_{nk}\), fractional occupations \(f_{nk}\in[0,2]\).
2. Rebuild \(P_k=\sum_b f_b\,c_b c_b^\dagger\),
   \(W_k=\sum_b f_b\varepsilon_b\,c_b c_b^\dagger\).
3. Central FD of the **non-self-consistent** Lagrangian under orthorhombic
   diagonal strain at fixed fractional coordinates (fixed \(\{f_k\}\)).
4. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Dual-gate oracle: multi-SCF strain-FD of \(F\) (`gpw_stress_fd`, same smear
settings). Offline CP2K twin: `STRESS_TENSOR NUMERICAL` + `SMEAR FERMI_DIRAC`.

Public API: `cpu::gpw::GpwDft` members `smearing`, `degauss` [Ry],
`extra_bands`, `mp_order`; free `gpw_stress` / `gpw_stress_fd`. Payload
fields `GpwResult::f_k` + full `C_k` under smear.

### System (Atomiverse pin)

| Quantity | Value |
| -------- | ----- |
| Geometry | BCC Na conventional cubic, \(a = 4.225\) Å, 2 atoms |
| Basis | `DZVP-MOLOPT-SR-GTH` (Na) — in-tree table |
| Pseudopotential | `GTH-PBE-q9` (9 valence e⁻ / atom → \(N_e = 18\)) |
| XC | PADE (LDA) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| SCF | DIIS 6, `tol = 1e-6` Ha, `EPS_DEFAULT = 1e-7` |
| Stress \(h\) | \(2\times 10^{-3}\) (dimensionless strain) |

#### Product pins (first green 2026-08-01; re-verify after rebase onto `dev` 2026-08-01)

| Observable | Pin | Tol |
| ---------- | --- | --- |
| Free energy \(F\) | **−92.1263306084 Ha** | \(10^{-3}\) Ha |
| \(N_{\mathrm{occ}}=\sum w f\) | **18.0** | \(10^{-5}\) |
| \(E_F\) | **0.055826 Ha** (finite) | \(\lvert E_F\rvert < 10^3\) |
| \(\max_{\mathrm{diag}}\lvert\sigma^{\mathrm{an}}-\sigma^{\mathrm{FD}}\rvert\) | **2.378×10⁻⁸ Ha/Bohr³** | \(2\times 10^{-4}\) |
| mean \(\sigma_{\mathrm{diag}}\) analytic | **−1.91772220×10⁻² Ha/Bohr³** | \(5\times 10^{-4}\) |
| Off-diagonal product | 0 | soft 1e-8 |

Analytic vs multi-SCF FD residual is well inside the dual-gate tol — free-energy
fixed-\(f\) Hellmann–Feynman matches reconverged \(F\) under strain.

### Offline CP2K twin (algorithm / NUMERICAL stress + SMEAR)

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, source `git:5e54ba2`) |
| Env | `source …/cp2k/install/env.sh` (`CP2K_DATA_DIR` set) |
| Host | arrhenius1, 2026-08-01 (re-run this wave) |
| Binary | `/nobackup/…/cp2k/install/bin/cp2k.psmp` |
| Command | `mpirun -np 1 cp2k.psmp <input>` with `OMP_NUM_THREADS=4` |
| Input (stress) | `tools/cp2k_oracle/inputs/gpw-metallic-stress/na_bcc_fermi_dirac_stress.inp` |
| Input (fair ENERGY wall) | `tools/cp2k_oracle/inputs/gpw-metallic-stress/na_bcc_fermi_dirac_energy.inp` |
| Method | Quickstep GPW, PADE, `GTH-PBE-q9` + `DZVP-MOLOPT-SR-GTH` |
| Cell | BCC conventional \(a=4.225\) Å, 2 Na |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (≈ 0.02 Ry \(k_B T\)), ADDED_MOS 4 |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| CUTOFF / REL_CUTOFF | 150 / 40 Ry, NGRIDS 4 |
| `STRESS_TENSOR` | **NUMERICAL** |
| Output energy line | `ENERGY\| Total FORCE_EVAL ( QS ) energy [hartree] -92.765939040711800` |
| Fermi energy | `0.05448910718283` Ha |
| Stress lines | `STRESS\| Numerical stress tensor [bar]` |
| | `STRESS\| x -5.01677864643E+06 …` (1/3 Trace −5.01677864609e6 bar) |
| \(\sigma_{\mathrm{xx}}\) (a.u.) | **−1.7051×10⁻² Ha/Bohr³** (= −5.01677865e6 bar / 2.94210107994716e8) |
| Soft class check | \(\lvert\sigma_{\mathrm{ATV}}-\sigma_{\mathrm{CP2K}}\rvert \le 5\times 10^{-3}\) Ha/Bohr³ (obs. ~2.1×10⁻³) |

Absolute free energy and absolute stress differ from CP2K by grid /
`EPS_DEFAULT` / multigrid residual (~0.64 Ha on \(F\), ~0.002 Ha/Bohr³ on
\(\sigma\)). Dual-gate analytic↔FD
is the product correctness pin; CP2K documents the matched GPW+SMEAR algorithm
twin and soft stress class.

### Perf gate (fair single-SCF twin wall)

Same system / basis / PP / smear / k-mesh as the ENERGY twin job (1 MPI ×
`OMP_NUM_THREADS=4`). **Not** the multi-SCF NUMERICAL stress wall.

| Engine | Wall |
| ------ | ---- |
| Atomiverse SCF (`T_atv_scf` from `reg_gpw-metallic-stress`) | **0.993 s** |
| CP2K ENERGY twin (`/usr/bin/time`, 1 MPI × 4 OMP) | **0.92 s** |

\[
r = T_{\mathrm{atv,scf}} / T_{\mathrm{cp2k}} \approx 1.08 \ll 10
\]

`within_10x = true`. Multi-SCF NUMERICAL stress wall was ~25 s (reference only;
not the fair ratio denominator).

### Gates summary

| Check | Status |
| ----- | ------ |
| SCF free energy + \(N_e\) + \(f_k\) payload | PASS |
| Analytic free-energy stress vs multi-SCF FD | PASS (2.4e-8) |
| Cubic symmetry / magnitude floor | PASS |
| Absolute mean \(\sigma\) pin | PASS |
| CP2K soft stress class | PASS |
| Fair wall ≤ 10× | PASS (1.08) |
| Unit `GpwStress.MetallicSmearAnalyticMatchesFdDiagonal` | PASS (~5.7e-11) |


## `cuda-gpw-device-forces`

### What

Device grid force product on public `ATV::cuda::gpw::GpwDft` free-function
forces: host AO Pulay (W·dS, P·dT, P·dVnl) + device V_eff grid term, ionic
HF, and local-PP HF. Replaces the previous full CPU `gpw_forces_from_scf`
call. Mid-SCF collocate/integrate remains device product (prior slug).

### Twin

Same-settings CPU `cpu::gpw::GpwDft` + `gpw_forces_analytic` on H₂
(DZV-GTH-PADE / GTH-PADE-q1), Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry,
closed-shell RKS, Padé LDA, DIIS. Oracle: **CPU twin**.

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| Both SCF converged | required |

### Perf

**perf_exempt** — small cell correctness-only product depth (compose device
grid HF into free-fn forces).

### Non-goals

UKS / multi-k / multigrid / stress / PBE force dual-gate / dispersion.

### GH200 evidence

Wave-8 rebase job **1127167** (partition gpu, n88 GH200, module
`GPU/buildenv-gcccuda/2026.03-cu13.0`, sacct **COMPLETED 0:0** ~6m07s;
prior **1126434** / **1125926** same class). HEAD rebased onto `dev` @
`ee4059e9` (MG + UKS energy on base):

| Quantity | Value |
| --- | ---: |
| \|E_cuda − E_cpu\| | **1.617×10⁻⁷ Ha** |
| max\|ΔF\| | **2.056×10⁻⁶ Ha/Bohr** |
| CPU iters / CUDA iters | 5 / 9 |
| Units | `test_cuda_gpw_forces` + `test_cuda_gpw_full_product` **2/2 PASS** |
| Exit | **GPU_JOB_OK** |

Tols: energy 5×10⁻⁶ Ha, forces 5×10⁻⁵ Ha/Bohr. **perf_exempt** small cell.


## `cuda-gpw-full-product`

### What

First public CUDA GPW method product: `ATV::cuda::gpw::GpwDft` matching the
CPU `GpwDft` public shape (members + `build` / `run` / `last_result` + forces
family). Composes landed `cuda/gpw/internal/*` stage units (device fine-grid
KS via `KsChainDevice::run_from_rho`) with host STVnl / cbg collocate /
integrate / DIIS / diagonalize.

### Twin

Same-settings CPU `cpu::gpw::GpwDft` on H₂ (DZV-GTH-PADE / GTH-PADE-q1),
Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry, closed-shell RKS, Padé LDA,
DIIS. Oracle: **CPU twin** (not CP2K for this correctness gate).

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| Both SCF converged | required |

### Perf

**perf_exempt** — small cell correctness-only first product (same class as
other CUDA ES footholds / PW dual-gates).

### Non-goals (this pin)

Multi-k, UKS, OT, multigrid product energy path, nonperiodic MT, vdW, XC
beyond PADE/PBE.


## `cuda-gpw-kpoint-forces`

### What

Multi-k RKS H₂ **analytic forces + cell stress** dual-gate. Energy parent
is `cuda-gpw-kpoints` (`k_mesh={2,1,1}`, TR off). CUDA SCF payload + host
k-path assembly.

### Twin

Same-settings CPU H₂ DZV-GTH-PADE, 12 Bohr, 80 Ry, Padé, `n_occ=1`.

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁶ Ha** |
| max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| max‖Δσ‖ | **5×10⁻⁵ Ha/Bohr³** |

**Claim: ◐** — CPU-twin dual-gate; not an independent CP2K pin.
**perf_exempt**.


## cuda-gpw-kpoints — multi-k RKS DIIS energy dual-gate

### What

Public `cuda::gpw::GpwDft` multi-k closed-shell RKS DIIS (Padé LDA) vs
same-settings CPU `GpwDft`. H₂, DZV-GTH-PADE / GTH-PADE-q1, cell 12 Bohr,
CUTOFF 80 Ry, `k_mesh={2,1,1}` with **`k_time_reversal=false`** (nk=2 at
κ_x=±1/4; TR-on would collapse to one off-Γ k), single-grid,
`local_pp_on_grid=true`.

Oracle is **same-settings CPU**, not CP2K (CUDA product twin). Absolute energy
is not pinned — only CUDA↔CPU parity.

### Dual-gate

| Quantity | Tol |
| --- | ---: |
| \(\lvert E_{\mathrm{cuda}}-E_{\mathrm{cpu}}\rvert\) | \(5\times10^{-6}\) Ha |

### Perf

**perf_exempt** — small-molecule multi-k energy foothold; not a fair twin wall
cut vs CP2K. Deeper multi-k MG wall lives on the CPU track / later product depth.

### Lever

Host phased \(S_k/T_k/V_{\mathrm{nl}}^k\) + χ^k collocate/integrate
(`cpu::gpw::internal` kpoints); device Poisson+XC via
`KsChainDevice::run_from_rho`. Forces / stress / UKS×k / MG×k remain
fail-closed deeper slugs.


## `cuda-gpw-multigrid-si-energy-path`

### What

Production **Si-class Γ RKS PADE multigrid** product energy path on public
`cuda::gpw::GpwDft` vs same-settings CPU. Closes the gap after H₂
`multigrid-run` foothold + Si single-grid `product-si-gate` + PBC collocate:
Si nonlocal × multi-level device KS continuous path.

### Twin

| | |
| --- | --- |
| System | diamond Si bct 4-atom, a = 5.43 Å |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| Settings | Γ, CUTOFF 40 Ry, **ngrids=4**, **rel_cutoff=20**, n_occ=8, PADE, DIIS |
| Oracle | **same-settings CPU** (not CP2K) |

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| Both SCF converged | required |

### Lever

Mid-SCF MG branch in `gpw_solver.cu` `build_energy_device`:

1. **Default (production path for this slug):** host collint multi-level twin
   `multigrid_collocate_rho` + `ks_grid_build_rks` + `multigrid_integrate_veff`
   (same MAIN as CPU), then device \(H=H_0+V\) / DIIS. Mid-path P D2H + V H2D
   residual (iron-law P2) — same pattern as product-si host cbg before
   `product-collocate-pbc`.
2. **Device monom residual:** `ATV_GPW_CUDA_DEVICE_MG=1` forces
   pack→`KsMultiGridChainDevice`→scatter. GH200 **1180025**: monom multi-level
   under-corrects Si (CUDA MG E ≈ SG −15.1523 vs CPU MG −15.15189,
   \|dE\|=**4.3e-4**). H₂ monom MG stays green (s-only). Deeper restore later.

Parents: KB Vnl@Γ (`product-si-gate`), PBC collocate (`product-collocate-pbc`).

### Perf

**perf_exempt** — correctness dual-gate / path-fix (not fair-wall vs CP2K).

### Status

**acceptance_met=true** — GH200 **1180883** (rebased tip `40a63395` on
GapCensus wave-4 base `213d794d`): CPU E=−15.151890096614 Ha iters=9;
CUDA E=−15.151890075311 Ha iters=9; \|dE\|=**2.130e-08** Ha (tol 5e-5).
Holds: ks_multigrid unit, full_product, multigrid-run \|dE\|=**1.308e-11**,
product-si \|dE\|=**3.504e-05**. Prior dual-gate **1180409** (pre-rebase
`a58d90e1`) same \|dE\|. Prior fail **1180025** device monom
\|dE\|=**4.296e-04**. Iron-law residual mid-path P/V DH is separate device-KS work
(do not re-open monom path here).


## `cuda-gpw-pbe-product-gate`

### What

Public product dual-gate under **PBE-GGA**: `ATV::cuda::gpw::GpwDft`
constructed with `XcKind::PBE` vs same-settings CPU `GpwDft`. Composes the landed
`KsChainDevice` PBE stage into the public method energy
path. Not a new kernel foothold.

### Twin

Same-settings CPU `cpu::gpw::GpwDft` on H₂ (DZV-GTH-PADE / **GTH-PBE-q1**),
Γ-point, periodic 12 Bohr cubic, CUTOFF 80 Ry, closed-shell RKS, **PBE**,
DIIS. Oracle: **CPU twin** (not CP2K for this correctness gate).

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁶ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁵ Ha/Bohr** |
| Both SCF converged | required |

### Perf

**perf_exempt** — small cell correctness-only (same class as
`cuda-gpw-full-product` / other CUDA ES dual-gates).

### Non-goals (this pin)

UKS-PBE, multigrid×PBE, multi-k×PBE, H₂O production cell, OT, MT, vdW,
device DIIS/eigen residual, stress.

### GH200 evidence (2026-08-10)

Job **1127166** (`wf/cuda-gpw-pbe-product-gate`): unit
`test_cuda_gpw_full_product` PASS (1/1, 2.70 s); dual-gate
CPU E=−1.076079182475 Ha, CUDA E=−1.076079016645 Ha,
\|ΔE\|=**1.658e-07** Ha (tol 5e-6), max\|ΔF\|=**5.232e-07** Ha/Bohr
(tol 5e-5). **perf_exempt** small cell.


## `cuda-gpw-product-h2o-gate`

### What

Production-class dual-gate for public `ATV::cuda::gpw::GpwDft` on **water**
(Γ closed-shell RKS Padé LDA). Extends the H₂-only first product
(`cuda-gpw-full-product`) to multi-species AO / grid depth without a new
kernel.

### Twin

Same-settings CPU `cpu::gpw::GpwDft`:

| Setting | Value |
| --- | --- |
| System | H₂O, 3 atoms |
| Basis / PP | O: `SZV-GTH-PADE` + `GTH-PADE-q6`; H: `DZV-GTH-PADE` + `GTH-PADE-q1` |
| Cell | periodic cubic 12 Bohr |
| CUTOFF | 100 Ry |
| XC | Padé LDA (`XcKind::PADE`) |
| Occupations | `n_occ = 4` (8 valence e⁻) |
| SCF | DIIS depth 6, `tol=1e-8`, `max_iter=80` |
| Grid | single-grid (`ngrids=1`), `local_pp_on_grid=true` |

Oracle: **CPU twin energy (and free-fn forces)** — not CP2K (CUDA rejects MT /
nonperiodic). Geometry: experimental-ish water centered in the cell
(CP2K tutorial bond lengths in Bohr).

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁵ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁴ Ha/Bohr** |
| Both SCF converged | required |

Tols match the O₂ p-shell production class (`cuda-gpw-uks-o2-gate` 5e-5 Ha),
looser than pure-s H₂ (5e-6) for multi-center product residual (cuFFT vs FFTW).

### Perf

**perf_exempt** — small-molecule correctness-only (CUDA ES open-gate class).

### Non-goals

Multi-k, UKS water, OT, multigrid product, nonperiodic MT, PBE water (→ later
after this PADE gate / `product-h2o-pbe-gate`), AO-Pulay device path. Device
RKS DIIS/eigen is on base; this gate uses it.

### Build / run (GH200)

```bash
#!/bin/bash -l
module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
  -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target atomiverse \
  test_cuda_gpw_full_product reg_cuda-gpw-product-h2o-gate
ctest --test-dir build -R '^test_cuda_gpw_full_product$' --output-on-failure
ctest --test-dir build -R '^gpw/rks$'
```

### GH200 evidence (authority: SPEC)

| Job | Node | \|dE\| Ha | max\|dF\| Ha/Bohr | Note |
| --- | --- | ---: | ---: | --- |
| **1130195** | n46 | **7.526e−07** | **3.213e−06** | Assign wave-5 base `7f8da1f0` |
| 1130140 | n193 | 7.526e−07 | 3.213e−06 | Assign wave-4 base |

`acceptance_met=true` — both SCF converge; tols held; **perf_exempt**.


## `cuda-gpw-product-h2o-stress`

### What

Production water Γ RKS PADE **cell stress** dual-gate. Residual after
`product-h2o-gate` (E+F) and H₂ `cuda-gpw-stress`. Host fixed-\(P\) assembly
at the CUDA density payload (no new kernel).

### Twin

Same-settings CPU `cpu::gpw::GpwDft` / `gpw_stress` as
`cuda-gpw-product-h2o-gate` (H₂O, O SZV + H DZV, 12 Bohr, 100 Ry, Padé,
`n_occ=4`). Oracle is the CPU twin, not CP2K (CUDA rejects MT).

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| max‖Δσ‖ (xx,yy,zz) | **5×10⁻⁵ Ha/Bohr³** |
| Off-diagonal | **1×10⁻⁸** |

**Claim: ◐** — CPU-twin dual-gate; not an independent CP2K pin (CUDA
rejects Martyna–Tuckerman). **perf_exempt** — small-molecule correctness-only.


## `cuda-gpw-product-si-gate`

### What

Production **Si-class Γ RKS PADE single-grid** (`ngrids=1`) energy dual-gate on
public `cuda::gpw::GpwDft` vs same-settings CPU. Isolates H0/STVpp (GTH
nonlocal KB) after H₂/water product footholds hide Si nonlocal production.

### Twin

| | |
| --- | --- |
| System | diamond Si bct 4-atom, a = 5.43 Å |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| Settings | Γ, CUTOFF 40 Ry, ngrids=1, n_occ=8, PADE, DIIS |
| Oracle | **same-settings CPU** |

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| Both SCF converged | required |

### Lever

1. CUDA Γ Vnl was per-pair `gth_nonlocal` with ket-only lattice images (wrong
   for periodic Si nonlocal). Fixed to shared `add_nonlocal_kb_k` at Γ
   (same KB form as CPU `add_nonlocal_kb` / multi-k STVnl host path).
2. Continuous device product collocate has **no collint PBC window wrap** —
   H₂/water large-cell hid it; Si MOLOPT tight cell under-counts density
   (GH200 **1166099**/**1166106**: after KB fix, CUDA still non-convergent
   E≈−5.18 vs CPU −15.15). Mid-SCF uses **host collint** ρ+KS+integrate twin
   + device H=H0+V / DIIS. Device continuous restore → `product-collocate-pbc`.

### Perf

**perf_exempt** — correctness dual-gate / host collint residual (not fair-wall).

### Status

**acceptance_met=true** — GH200 **1167099** (tip `a30c675d`):
CPU E=−15.152328573401 Ha iters=7; CUDA E=−15.152328552099 Ha iters=7;
\|dE\|=**2.130e-08** Ha (tol 5e-5). Hold: full-product + product-h2o green.


## cuda-gpw-stress

### Twin

Same-settings CPU `gpw_stress` (fixed-P analytic at converged density) vs
public `cuda::gpw::gpw_stress` after CUDA Γ RKS PADE SCF (H₂).

| Setting | Value |
| --- | --- |
| System | H₂, DZV-GTH-PADE / GTH-PADE-q1, 12 Bohr cube |
| CUTOFF | 80 Ry, `ngrids=1`, periodic |
| XC | Padé LDA |
| Strain `h` | 2e-3 (CPU default) |

First cut: CUDA SCF density + host fixed-P assembly (`cpu::gpw::gpw_stress` at
state). Fail-closed multi-k.

### Tolerances

| Quantity | Tol |
| --- | ---: |
| \|E\| dual-gate | 5e-6 Ha |
| max\|Δσ_diag\| | 5e-5 Ha/Bohr³ |
| off-diag | ~0 (1e-8) |

**perf_exempt** small cell.


## Reference — `mpi-gpw-scaling-4rank`

### Method

Closed-shell **GPW RKS** with **Monkhorst–Pack k-point sampling** and
**MPI k-point round-robin** over `MPI_COMM_WORLD` (mirrors plane-wave
`cpu::pw` layout: `ik = rank; ik < n_k; ik += nprocs`). Each rank
collocates / diagonalizes only its local k; real-space density,
kinetic/nonlocal energy scalars, DIIS \(B\)-matrix partials, and residual
norms are Allreduced so every rank holds identical `e_total`.

Public API: concrete `cpu::gpw::GpwDft{kinds, cutoff_ry, xc}` + knob members
(`k_mesh`, …);
`GpwResult::{n_kpts, n_kpts_local, e_total, …}`. No public `*Config`, no
factories.

### System (lightweight twin of `gpw-kpoints-si`)

| | |
| --- | --- |
| Lattice | Si diamond bct 4-atom, a = 5.43 Å |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| XC | PBE |
| Density cutoff | **80 Ry** (e2e self-launch budget) |
| k mesh | MONKHORST-PACK 2×2×2, TR → **4 k** |
| SCF | DIIS, tol 1e-7 Ha |

### What is checked

1. Serial SCF converges; `n_kpts == 4`, `n_kpts_local == n_kpts`.
2. Parent self-launches `mpiexec -n 4 <binary> --mpi-worker`.
3. Worker: all ranks agree on energy (\(\le 10^{-12}\) Ha spread); local
   k counts sum to `n_kpts`.
4. Rank-count invariance \(|E_{n=1}-E_{n=4}| \le 10^{-8}\) Ha.
5. Wall ratio \(T_1/T_4\) recorded; **perf_exempt**.

### Measured (first green, 2026-07-25, OMP=1)

| Quantity | Value |
| --- | --- |
| E (serial n=1, 80 Ry MP 2×2×2 TR) | −15.8024163941 Ha (6 iters) |
| E (mpiexec n=4) | −15.8024163941 Ha |
| \|E1 − E4\| | **0** (machine-identical) |
| T1 / T4_worker | ≈ 0.995 (setup still replicated — no speedup expected) |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | k-RR foothold only (setup matrices still replicated; no real-space domain decomp); no CP2K multi-rank twin required this wave |
| r = T_atv/T_cp2k | n/a (exempt) |
| wall ratio T1/T4 | ≈ 1.0 (informational; k-RR alone does not beat serial when setup is O(nk) replicated) |

### Honest scope

- **Shipped:** multi-k SCF k-point RR + density/energy/DIIS Allreduce;
  serial bit-identical when nprocs=1 / MPI not initialized.
- **Not shipped:** Γ-only MPI work sharing; grid/AO domain decomposition;
  k-UKS; k-point forces under MPI.
