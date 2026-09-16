# GPW PBE-D3 (zero damping) analytic cell stress — periodic water dimer

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d3-stress` |
| Regression | `regressions/gpw/gpw-d3-stress/` |
| Public API | `cpu::gpw::GPWEvaluator` (`d3`, `compute_stress`, `stress_use_fd=false`); free `gpw_stress` / `gpw_stress_fd` with `GpwDft::d3` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

Gates **Hellmann–Feynman fixed-\(P\) orthorhombic cell stress** on
**Γ-point GPW PBE** with additive **DFT-D3 zero ("original") damping**
(two-body, `CALCULATE_C9_TERM .FALSE.`) for a **periodic water dimer**
(same 12 Å cubic cell as the `gpw-d3-forces` periodic twin and
`gpw-d4-stress`). Dispersion is geometry-only; the analytic path rebuilds
\(E_{\mathrm{disp}}^{\mathrm{D3}}(\{\mathbf{R}\},\mathbf{h})\) at each
strained cell while holding the unstrained density matrix \(P\) and
energy-weighted \(W\), using the CP2K-parity `kD3Cp2kPairCutoffBohr`
(= 2 × `R_CUTOFF 20.0` Å) lattice-image cutoff.

**Oracle (binding).** Unlike `gpw-d4-stress` (Arrhenius CP2K lacks DFTD4),
CP2K 2026.1 **does** run `TYPE DFTD3`, so this slug carries a real
**CP2K `STRESS_TENSOR NUMERICAL` difference pin**: two offline runs on the
identical geometry — with and without the `&VDW_POTENTIAL` block — whose
stress difference isolates the D3-zero dispersion stress exactly (D3 does
not enter the Kohn–Sham potential, so the electronic stress cancels).
Decisive internal residual: product analytic vs multi-SCF strain-FD
(`gpw_stress_fd`) on the same `d3=true` method.

Public surface: bool / stress members on `cpu::gpw::GPWEvaluator` and
`cpu::gpw::GpwDft` (no `*Config` bag, no `make_*` / `build_*`). Frame
`kStress` is eV/Å³; pin channel `last_stress_ha_bohr3()` [Ha/Bohr³].
Product gate: `regressions/gpw/gpw-d3-stress` (`reg_gpw-d3-stress`).

## Method

### Host + dispersion

Closed-shell **Γ-point GPW** Kohn–Sham (VandeVondele et al., *Comput. Phys.
Commun.* **167**, 103 (2005)) with PBE XC and GTH PBE PPs. **DFT-D3 zero
damping** (Grimme, Antony, Ehrlich, Krieg, *J. Chem. Phys.* **132**, 154104
(2010)) with PBE reference parameters:

| Parameter | Value |
| --------- | ----: |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.722 |
| \(s_{r,6}\) | 1.217 |
| \(s_{r,8}\) | 1.0 |

Periodic lattice sums to `kD3Cp2kPairCutoffBohr` = 75.589 Bohr (the CP2K
half-cutoff convention that the `gpw-d3-forces` pin settled; on this
**periodic** cell the convention moves the pin, unlike the isolated twin).

### Analytic fixed-\(P\) cell stress with D3

One SCF at the unstrained cell → \(P\), \(W\). Central FD of the
non-self-consistent Lagrangian under orthorhombic diagonal strain at fixed
fractional coordinates:

\[
E_L[P,W;\varepsilon]
  = \mathrm{Tr}(P\,T) + \mathrm{Tr}(P\,V_{\mathrm{pp}})
  + E_{\mathrm{es}}[\rho_P] + E_{\mathrm{xc}}[\rho_P]
  + E_{\mathrm{disp}}^{\mathrm{D3}}(\{\mathbf{R}\},\mathbf{h})
  - \mathrm{Tr}(W\,S),
\qquad
\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}.
\]

Sign matches CP2K `STRESS_TENSOR`; Voigt order xx,yy,zz,xy,xz,yz;
off-diagonals ~0 (orthorhombic); \(h = 2\times10^{-3}\) (`stress_strain_h`).

**Relation to other D3 / stress slugs.**

| Slug | Deliverable |
| ---- | ----------- |
| [`gpw-d3-forces`](gpw-d3-forces.md) | D3-zero **forces** (isolated) + periodic E_disp pin + cutoff convention |
| [`gpw-d4-stress`](gpw-d4-stress.md) | D4 stress, dual-gate only (no CP2K DFTD4 build) |
| [`gpw-mbd-stress`](gpw-mbd-stress.md) | Γ MBD stress |
| **`gpw-d3-stress`** | D3-zero stress with a real **CP2K stress-difference pin** |

**Key exclusions.** D3(BJ) stress (`gpw-d3bj-stress`, sibling row); ATM/C9
stress; shear / triclinic; multi-k dedicated pin (the multi-k fold shares
the same `e_disp` branch); CUDA GPW.

## Implementation

Public surface only (`GPWEvaluator` / `GpwDft` members; Frame in/out):

```cpp
cpu::gpw::GPWEvaluator eval({O, H});
eval.cutoff_ry = 200.0;
eval.tol = 1e-7;             // matches CP2K EPS_SCF twin
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = false;    // PERIODIC XYZ
eval.d3 = true;              // zero damping; exclusive vs d3bj/d4/ts/xdm/mbd
eval.compute_stress = true;  // product analytic fixed-P
eval.stress_use_fd = false;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

const double e_disp = eval.last_result().e_disp;   // [Ha]
const auto& san = eval.last_stress_ha_bohr3();     // [Ha/Bohr³]
// frame.meta.get(kStress)                          // [eV/Å³]
```

The regression isolates ATV's dispersion stress the same way the pin does:
a second warm-started analytic product with `d3=false`, subtracted.

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `d3`, `compute_stress`, `last_stress_ha_bohr3` |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::d3`, `gpw_stress` / `gpw_stress_fd` |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Fixed-\(P\)+Pulay Lagrangian; D3 rebuilt under strain (Γ + multi-k) |
| `include/atomiverse/cpu/gpw/internal/dftd3.h` | `dftd3_pbe`, `kD3Cp2kPairCutoffBohr` |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | `AnalyticMatchesFdWithD3`, `D3ChangesGammaDiagonal` |
| `regressions/gpw/gpw-d3-stress/` | Product gate + REFERENCE.md pins |
| `tools/cp2k_oracle/inputs/gpw-d3-stress/` | CP2K NUMERICAL stress twin inputs |

## Validation (CP2K 2026.1 + dual-gate)

Oracle: **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`,
git:5e54ba2), run **offline** on the Arrhenius epyc login node
(`mpirun -np 2`, `OMP_NUM_THREADS=2`), 2026-07-31 — never launched from
ctest. Inputs: `tools/cp2k_oracle/inputs/gpw-d3-stress/`
(`water_dimer_d3_stress.inp`, `water_dimer_plain_stress.inp`). Full
provenance and re-run recipe:
[`regressions/gpw/gpw-d3-stress/REFERENCE.md`](../../../regressions/gpw/gpw-d3-stress/REFERENCE.md).

CP2K printed (Hartree): total (plain) −34.338293290245112,
total (D3) −34.339114184921726, `Dispersion energy`
**−0.00082089467662**. The NUMERICAL stress diagonals (bar; converted at
1 Ha/Bohr³ = 2.94210107994716×10⁸ bar) give the hard difference pin
σ_num(D3) − σ_num(plain).

### Accuracy (pins vs measured, `reg_gpw-d3-stress` first green 2026-07-31)

ATV: 11 SCF iters, E_tot = −34.331152296855 Ha (single 200 Ry grid vs
CP2K NGRIDS 4 / REL_CUTOFF 40 — geometry-only pins are cutoff-independent).

| Gate | CP2K pin | Tol | ATV measured |
| ---- | -------- | --- | ------------ |
| \(E_{\mathrm{disp}}\) [Ha] | **−0.00082089467662** | 1×10⁻⁸ | −8.20894731919×10⁻⁴ (\|Δ\| = 5.53×10⁻¹¹) |
| Δσ_xx [Ha/Bohr³] | **−1.70641140246×10⁻⁸** | 2×10⁻⁹ | \|Δ\| = 2.27×10⁻¹⁰ |
| Δσ_yy [Ha/Bohr³] | **−9.03962616420×10⁻¹⁰** | 2×10⁻⁹ | \|Δ\| = 2.26×10⁻¹² |
| Δσ_zz [Ha/Bohr³] | **+3.74278507120×10⁻⁹** | 2×10⁻⁹ | \|Δ\| = 1.92×10⁻¹¹ |
| max\|σ_an − σ_fd\| diag (internal FD twin, d3=true) | — | 2×10⁻⁵ | 7.39×10⁻⁷ (xx; yy 2.46×10⁻⁸, zz 1.24×10⁻⁹) |
| max\|σ_tot − σ_cp2k\| diag (sanity: single-grid vs multigrid) | soft: (1.85034×10⁻⁵, −3.62033×10⁻⁷, 4.35412×10⁻⁵) | 1×10⁻⁴ | 4.02×10⁻⁵ |
| off-diagonal product | 0 (orthorhombic) | 1×10⁻⁸ | ~0 |

Product analytic diagonals [Ha/Bohr³]: xx −2.16997707×10⁻⁵,
yy +2.00723305×10⁻⁵, zz +5.82684354×10⁻⁵.

The Δσ residual (2.27×10⁻¹⁰ on xx) is the SCF-noise floor of the
two-full-products isolation (both analytic runs re-converge to `tol = 1e−7`);
signal separation stays decisive — xx signal 1.71×10⁻⁸ = 75× the residual,
tolerance 2×10⁻⁹ keeps 8.8× headroom.

## Performance

Fair comparison: same observable (this cell's stress). CP2K uses NUMERICAL
strain-FD (many SCFs); ATV's product path is one SCF + fixed-\(P\) analytic
stress. Settings caveat: ATV single 200 Ry grid, OMP 8; CP2K NGRIDS 4,
np 2 × OMP 2 (Arrhenius epyc).

| Run | Wall [s] |
| --- | -------: |
| CP2K NUMERICAL stress, D3 | 33.6 |
| CP2K NUMERICAL stress, plain | 38.9 |
| ATV analytic product (SCF + fixed-P stress, OMP 8) | **18.5** |
| ATV plain twin (difference pin only) | 14.9 |
| ATV all-axes strain-FD twin (dual gate only) | 14.3 |
| Regression total (≤ 60 s budget) | 47.7 |

**Ratio (ATV product / CP2K D3 twin) = 18.5 / 33.6 ≈ 0.55**, inside the
10× fair-wall budget. The extra plain/FD twins exist only for the
difference pin and dual gate, not the product path.

## References

- Offline pin: `regressions/gpw/gpw-d3-stress/REFERENCE.md`
- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**,
  154104 (2010) (DFT-D3, zero damping)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) (GPW)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) (stress context)
- Sibling pages: [gpw-d3-forces](gpw-d3-forces.md),
  [gpw-d4-stress](gpw-d4-stress.md), [gpw-mbd-stress](gpw-mbd-stress.md)
