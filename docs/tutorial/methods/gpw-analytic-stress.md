# GPW analytic cell stress — H₂O / PBE / DZVP-MOLOPT-GTH (MT)

| | |
|---|---|
| Slug | `gpw-analytic-stress` |
| Public API | `cpu::gpw::GpwDft::stress` (analytic); oracle `gpw_stress_fd` [eV/Å³]; `GPWEvaluator` (`compute_stress`, `stress_use_fd=false` default) |
| Roadmap | [dft.md](../../roadmap/dft.md) § Periodic response; queue `gpw-analytic-stress` |

## What this is

**Hellmann–Feynman fixed-\(P\) cell stress** for closed-shell Γ-point GPW
(Quickstep): one SCF, then central FD of the non-self-consistent Lagrangian

\[
E_L[P,W;\varepsilon]
  = \mathrm{Tr}(P\,T) + \mathrm{Tr}(P\,V_{\mathrm{pp}})
  + E_{\mathrm{es}}[\rho_P] + E_{\mathrm{xc}}[\rho_P] + E_{\mathrm{disp}}
  - \mathrm{Tr}(W\,S)
\]

under orthorhombic diagonal strain at fixed fractional coordinates. The
\(-\mathrm{Tr}(W\,S)\) term is the Pulay contribution
(\(W = \sum_i 2\varepsilon_i c_i c_i^\top\)). Multi-SCF strain-FD remains
`gpw_stress_fd` for dual-gate only.

Sign: \(\sigma_{aa} = -\Omega^{-1}\partial E/\partial\varepsilon_{aa}\)
(CP2K `STRESS_TENSOR`). Frame `kStress` is eV (virial σ·Ω); pin compare via
`last_stress_ha_bohr3()` [Ha/Bohr³].

Product gate: analytic vs multi-SCF FD diagonals within **1×10⁻⁵** Ha/Bohr³
on the water MT twin (measured residual ~**3×10⁻¹⁰**). Absolute offline
CP2K NUMERICAL pin remains on `gpw-stress-water` (same product path now
defaults to analytic; residual ~10⁻⁸ Ha/Bohr³).

## Scope / non-goals

**In:** closed-shell Γ RKS; orthorhombic diagonal; `local_pp_on_grid` consistent
with SCF; optional D3/D3(BJ) geometry energy; axes mask.

**Out:** closed-form term tables without fixed-\(P\) samples; shear/triclinic;
UKS / k-points / DFT+U / D4; CP2K `STRESS_TENSOR ANALYTICAL` absolute pin on
nonperiodic MT (disagrees with NUMERICAL — see `gpw-stress-water`).

## Minimal sequence

```cpp
cpu::gpw::GpwDft dft{{O, H}, 400.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = true;
cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_stress = true;       // product analytic (default)
// eval.stress_use_fd = true;     // multi-SCF oracle only
eval.method().stress_strain_h = 2e-3;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

const auto& s = eval.last_stress_ha_bohr3(); // Ha/Bohr³ Voigt
// frame.meta.get(kStress) is eV (virial σ·Ω)
```

## Files

| Path | Role |
| ---- | ---- |
| `gpw_solver.h` | `GpwDft::stress` (`stress_strain_h`, `stress_axes`) / `gpw_stress_fd` |
| `gpw_evaluator.h` | `compute_stress`, `stress_use_fd` |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Fixed-\(P\)+Pulay + multi-SCF FD |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | Unit dual-gate |
| `gpw-analytic-stress` | Product twin |

## References

- Offline pins + deferred terms: pin notes
- Strain-FD / CP2K NUMERICAL history: [`gpw-stress-water`](gpw-stress-water.md)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985); Quickstep forces/stress (VandeVondele et al., CPC 167, 103 (2005))
