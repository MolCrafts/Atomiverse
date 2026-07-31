# GPW PBE-D3 three-body ATM (C9) analytic cell stress — periodic water dimer

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d3-atm-stress` |
| Regression | `regressions/gpw/gpw-d3-atm-stress/` |
| Public API | `cpu::gpw::GPWEvaluator` (`d3`, `d3_atm`, `d3_rcut_bohr`, `compute_stress`, `stress_use_fd=false`); free `gpw_stress` / `gpw_stress_fd` with `GpwDft::{d3, d3_atm, d3_rcut_bohr}` |
| Roadmap / queue | `docs/roadmap/dft.md` §8 Dispersion; `docs/developer/cp2k-method-queue.md` (worker wave 2 2026-07-31) |

## Summary

Pins the **three-body Axilrod–Teller–Muto (C9) contribution to the
Hellmann–Feynman fixed-\(P\) cell stress** on Γ-point GPW PBE against an
offline **CP2K `STRESS_TENSOR NUMERICAL` twin pair** — the stress sequel to
[`gpw-d3-atm-forces`](gpw-d3-atm-forces.md) and the ATM sibling of
[`gpw-d3bj-stress`](gpw-d3bj-stress.md). The pin is the **difference of two
CP2K runs** (`CALCULATE_C9_TERM` `.TRUE.` − `.FALSE.`, otherwise identical):
D3 never enters the KS potential, so the electronic **and two-body
dispersion** stress cancel exactly, isolating the pure ATM stress
\(-\Omega^{-1}\partial E_{\mathrm{ATM}}/\partial\varepsilon\).

Two things ship with the pin:

1. **`d3_rcut_bohr`** — the DFT-D3 pair/triple hard cutoff as a public
   member on `GpwDft` / `GPWEvaluator` (CP2K `&PAIR_POTENTIAL R_CUTOFF`,
   doubled convention, default = CP2K default 20 Å → 75.59 Bohr). Energy,
   forces and stress all fold the one configured value, so the product can
   match any CP2K D3 input exactly (this twin: `R_CUTOFF 10.0` → 37.79 Bohr).
2. The finding that the **ATM strain-FD is step-locked**: the triple sum
   passes hard `r < r_cut` and empirical-scaling gates, so its included
   boundary set jumps (~5e−11 Ha) at discrete strains. The regression's
   ATM-only FD therefore uses **exactly CP2K's cell-perturbation step**
   (`&NUMERICAL_STRESS` default `DX 0.001` Bohr ⇒ ε = DX/L): at h = 1e−4 the
   xx/zz parity degraded to 1.6e−10 Ha/Bohr³; at the CP2K step it is 1.1e−13.

## Method

Γ-point GPW (Quickstep), PBE XC, GTH-PBE PPs, DZVP-MOLOPT-SR-GTH, on the
periodic **6 Å** water-dimer cell of `gpw-d3-atm-forces`. DFT-D3 zero
damping (Grimme et al., *J. Chem. Phys.* **132**, 154104 (2010)) with the
ATM/C9 term as CP2K folds it (`qs_dispersion_d3`: \(s_9 = s_6\), damping
`damping_d3(rabc, r0, 4/3, 16)`, half-neighbor-list triple enumeration).

Product stress: one SCF → central FD of the non-self-consistent Lagrangian
\(E_L[P;W;\varepsilon]\) under orthorhombic diagonal strain at fixed
fractional coordinates, with the geometry-only
\(E_{\mathrm{disp}}^{\mathrm{D3+ATM}}\) rebuilt at each strained cell;
\(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\). The ATM
term contributes **no** `D3Result::virial` — the fixed-P path rebuilds the
energy instead, which is exactly the fold this pin certifies.

## Implementation

```cpp
cpu::gpw::GPWEvaluator eval({O, H});
eval.cutoff_ry = 150.0;
eval.tol = 1e-6;
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = false;
eval.d3 = true;              // zero damping; exclusive vs d3bj / d4 / …
eval.d3_atm = true;          // CP2K CALCULATE_C9_TERM
eval.d3_rcut_bohr = 2.0 * 10.0 * 1.8897259885789; // CP2K R_CUTOFF 10.0 twin
eval.compute_stress = true;  // analytic fixed-P (product default)
eval.stress_use_fd = false;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);
const auto& san = eval.last_stress_ha_bohr3();  // [Ha/Bohr³] Voigt
// frame.meta.get(kStress)                      // [eV/Å³]
```

| Path | Role |
| ---- | ---- |
| `src/cpu/gpw/internal/gpw_stress.cpp` | fixed-\(P\) Lagrangian + D3+ATM rebuild (`energy_fixed_P` / `energy_fixed_P_k`) |
| `src/cpu/gpw/internal/dftd3.cpp` | `accumulate_atm_c9` (CASE 1/2/3 image triples, pbc anchoring) |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::d3_rcut_bohr` (new member) |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | `GpwStress.AnalyticMatchesFdWithD3Atm` (cheap **periodic** H₂ — image triples engage) |
| `regressions/gpw/gpw-d3-atm-stress/` | product gate + REFERENCE.md pins |
| `tools/cp2k_oracle/inputs/gpw-d3-atm-stress/` | offline CP2K stress twin pair |

## Validation (offline CP2K 2026.1)

Source:
[`regressions/gpw/gpw-d3-atm-stress/REFERENCE.md`](../../../regressions/gpw/gpw-d3-atm-stress/REFERENCE.md).
Two runs (`mpirun -np 2`, OMP 2, `EPS_SCF 1e-8`, CUTOFF 300, per-6 Å cell):
`water_dimer_d3_atm_per6_stress.inp` / `water_dimer_d3_per6_stress.inp`.

| Quantity | CP2K pin | ATV | Tol |
| -------- | -------: | --: | --- |
| \(E_{\mathrm{disp}}\)(C9) [Ha] | −0.00195809486938 | \|d\| = 2.8e−10 | 1e−9 Ha |
| \(E_{\mathrm{disp}}\)(2b) [Ha] | −0.00195887984277 | \|d\| = 2.8e−10 | 1e−9 Ha |
| \(\Delta\sigma_{xx}\) [Ha/Bohr³] | −1.7148335571e−08 | \|d\| = 1.1e−13 | 5e−12 |
| \(\Delta\sigma_{yy}\) [Ha/Bohr³] | +4.3332828109e−08 | \|d\| = 1.1e−14 | 5e−12 |
| \(\Delta\sigma_{zz}\) [Ha/Bohr³] | +9.0917867267e−09 | \|d\| = 3.7e−14 | 5e−12 |
| dual gate \(|\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}|\) (xx in e2e) | — | 2.4e−7 | 2e−5 Ha/Bohr³ |

## Performance

| Twin | \(T\) [s] |
| ---- | --------: |
| CP2K numerical-stress C9 run (18 perturbed SCFs + ref, 2×2 cores) | 54.5 |
| ATV product analytic stress pass (1 SCF + 7 fixed-\(P\) evals, OMP 8) | 40.8 |
| ATV multi-SCF FD twin, x-only, warm-started (dual-gate only) | 7.8 |

Ratio \(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) ≈ **0.75** (≤ 10; loaded login
node — settings caveats in REFERENCE.md).

## Limitations

- Water-dimer 6 Å periodic cell — image-triple consistency pin, not bulk.
- Zero-damping D3 + ATM (the twin the force pin established); the
  D3(BJ)+ATM combination is unpinned (CP2K supports it; fold is shared).
- Orthorhombic diagonal only; shear/triclinic out of scope.
- ATM has no analytic virial; a virial-path consumer would be a new row.
- D4 ATM stress is the sibling queue row `gpw-d4-atm-stress` (no CP2K pin
  possible — Arrhenius CP2K lacks libdftd4).

## References

- Offline pin: `regressions/gpw/gpw-d3-atm-stress/REFERENCE.md`
- S. Grimme et al., *J. Chem. Phys.* **132**, 154104 (2010) (DFT-D3 + ATM)
- B. M. Axilrod, E. Teller, *J. Chem. Phys.* **11**, 299 (1943); Y. Muto,
  *Proc. Phys. Math. Soc. Jpn.* **17**, 629 (1943) (triple-dipole term)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
  (Quickstep GPW)
- Siblings: [gpw-d3-atm-forces](gpw-d3-atm-forces.md),
  [gpw-d3bj-stress](gpw-d3bj-stress.md), [gpw-d3-atm](gpw-d3-atm.md),
  [gpw-d4-atm-forces](gpw-d4-atm-forces.md)
