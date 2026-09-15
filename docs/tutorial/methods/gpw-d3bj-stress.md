# GPW PBE-D3(BJ) analytic cell stress — periodic water dimer

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d3bj-stress` |
| Regression | `regressions/gpw/gpw-d3bj-stress/` |
| Public API | `cpu::gpw::GPWEvaluator` (`d3bj`, `compute_stress`, `stress_use_fd=false`); free `gpw_stress` / `gpw_stress_fd` with `GpwDft::d3bj` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

Pins **Hellmann–Feynman fixed-\(P\) orthorhombic cell stress** on **Γ-point
GPW PBE** with additive **DFT-D3(BJ)** (rational damping, two-body, ATM off)
against a real **offline CP2K `STRESS_TENSOR NUMERICAL` twin** — the case the
D4 sibling could not carry (Arrhenius CP2K lacks libdftd4, so
[`gpw-d4-stress`](gpw-d4-stress.md) gates on the internal dual-gate only).
D3(BJ) is native in CP2K, which makes this slug the **oracle stress pin for
the whole geometry-only dispersion stress fold** in
`src/cpu/gpw/internal/gpw_stress.cpp`.

The stress pin is a **difference of two CP2K runs** (D3BJ − plain), same
construction as the [`gpw-d3bj-forces`](gpw-d3bj-forces.md) force pin: D3
never enters the KS potential, so at every strained cell the two SCF
densities are identical and the numerical-stress difference is exactly the
FD strain-derivative of \(E_{\mathrm{disp}}\) alone — electronic stress
*and* CP2K's FD truncation cancel.

## Method

Γ-point GPW (Quickstep; VandeVondele et al., *Comput. Phys. Commun.* **167**,
103 (2005)), PBE XC, GTH-PBE PPs, DZVP-MOLOPT-SR-GTH. **DFT-D3(BJ)**
(Grimme et al., *J. Comput. Chem.* **32**, 1456 (2011)) with CP2K's PBE
parameters and CN convention:

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.7875 | dimensionless |
| \(a_1\) | 0.4289 | dimensionless |
| \(a_2\) | 4.4407 | Bohr |
| pair cutoff | 2 × `R_CUTOFF 20 Å` = 75.589 | Bohr |

Product stress: one SCF → central FD of the non-self-consistent Lagrangian
\(E_L[P;W;\varepsilon]\) under orthorhombic diagonal strain at fixed
fractional coordinates, with
\(E_{\mathrm{disp}}^{\mathrm{D3(BJ)}}(\{\mathbf{R}\},\mathbf{h})\) rebuilt at
each strained cell; \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial
\varepsilon_{aa}\) (CP2K sign). Multi-SCF `gpw_stress_fd` is the internal
dual-gate twin, never the product default.

## Implementation

Public surface only — same shape as `gpw-d4-stress` with `d3bj` instead of
`d4`:

```cpp
cpu::gpw::GPWEvaluator eval({O, H});
eval.cutoff_ry = 200.0;
eval.tol = 1e-6;
eval.mix = 0.4;
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = false;
eval.d3bj = true;            // exclusive vs d3 / d4 / ts / xdm / mbd
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
| `src/cpu/gpw/internal/gpw_stress.cpp` | fixed-\(P\) Lagrangian + D3(BJ) rebuild (`energy_fixed_P` / `energy_fixed_P_k`) |
| `include/atomiverse/cpu/gpw/internal/dftd3.h` | `dftd3_pbe_bj`, `kD3Cp2kPairCutoffBohr` |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | `GpwStress.AnalyticMatchesFdWithD3Bj` (cheap MT H₂) |
| `regressions/gpw/gpw-d3bj-stress/` | product gate + REFERENCE.md pins |
| `tools/cp2k_oracle/inputs/gpw-d3bj-stress/` | offline CP2K stress twins |

## Validation (offline CP2K 2026.1)

Source:
[`regressions/gpw/gpw-d3bj-stress/REFERENCE.md`](../../../regressions/gpw/gpw-d3bj-stress/REFERENCE.md).
Two runs (`mpirun -np 2`, OMP 2, `EPS_SCF 1e-8`, CUTOFF 300):
`water_dimer_d3bj_stress.inp` and `water_dimer_plain_stress.inp`, both
`STRESS_TENSOR NUMERICAL`.

| Quantity | CP2K pin | ATV | Tol |
| -------- | -------: | --: | --- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.00137627769481 | \|d\| = 7.6e−12 | 1e−8 Ha |
| \(\Delta\sigma_{xx}\) [Ha/Bohr³] | −1.1485894562e−07 | \|d\| = 3.9e−14 | 1e−10 |
| \(\Delta\sigma_{yy}\) [Ha/Bohr³] | −8.9319862445e−10 | \|d\| = 2.7e−14 | 1e−10 |
| \(\Delta\sigma_{zz}\) [Ha/Bohr³] | +9.6563695465e−10 | \|d\| = 3.2e−14 | 1e−10 |
| dual gate \(|\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}|\) (xx in e2e) | — | 7.4e−7 (pinning all-axes: yy 2.5e−8, zz 1.2e−9) | 2e−5 Ha/Bohr³ |

## Performance

| Twin | \(T\) [s] |
| ---- | --------: |
| CP2K numerical-stress D3BJ run (18 perturbed SCFs + ref, 2×2 cores) | 83.8 |
| ATV product analytic stress pass (1 SCF + 7 fixed-\(P\) evals, OMP 8) | 28.4 |
| ATV multi-SCF FD twin, x-only, warm-started (dual-gate only) | 4.8 |

Ratio \(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) ≈ **0.37** (≤ 10; stress
product vs stress oracle — settings caveats in REFERENCE.md).

## Limitations

- Water-dimer periodic cell — lattice-sum consistency pin, not bulk ice.
- Two-body D3(BJ) only (`CALCULATE_C9_TERM .FALSE.`); ATM is
  `gpw-d3-atm-forces` territory.
- Orthorhombic diagonal only; shear/triclinic out of scope.
- Γ-RKS on this pin; the multi-k fold exists but is pinned elsewhere.
- Zero-damping D3 stress is the sibling queue row `gpw-d3-stress`.

## References

- Offline pin: `regressions/gpw/gpw-d3bj-stress/REFERENCE.md`
- S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011)
  (D3(BJ) damping)
- S. Grimme et al., *J. Chem. Phys.* **132**, 154104 (2010) (DFT-D3)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
  (Quickstep GPW)
- Siblings: [gpw-d3bj-forces](gpw-d3bj-forces.md),
  [gpw-d4-stress](gpw-d4-stress.md), [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md),
  [gpw-stress-water](gpw-stress-water.md)
