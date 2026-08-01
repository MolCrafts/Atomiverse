# Multi-k GPW DFT-D4 two-body (BJ) analytic ionic forces

*(Technical-report page; CP2K capability loop.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d4-kpoint-forces` |
| Regression | `regressions/gpw/gpw-d4-kpoint-forces/` |
| Public API | `cpu::gpw::GPWEvaluator` / `GpwDft`: `k_mesh = {1,1,2}`, `d4 = true`, `d4_atm = false`, `compute_forces = true` |
| Roadmap / queue | `docs/roadmap/dft.md` §8 Dispersion + §10 Periodic KS; `docs/developer/cp2k-method-queue.md` |
| Oracle | **CP2K 2026.1** multi-k `ENERGY_FORCE` (libdftd4 rebuild) + **dftd4 3.7.0** geometry dual |

## Summary

**Monkhorst–Pack multi-k GPW** ionic forces with additive **DFT-D4 two-body**
Becke–Johnson damping (ATM off, \(s_9 = 0\)). Γ already folds two-body D4 into
`assemble_gpw_forces` (`gpw-d4-forces`); this capability is the **dedicated
multi-k product pin** proving the same geometry-only
\(F_{\mathrm{disp}} = -\nabla E_2\) is added on `assemble_gpw_forces_kpoints`
when `d4` is set.

Product ionic force:

\[
\mathbf{F}_A^{\mathrm{tot}}
  = \mathbf{F}_A^{\mathrm{k\text{-}HF/Pulay}}
  + \mathbf{F}_A^{\mathrm{disp}},
\quad
\mathbf{F}_A^{\mathrm{disp}}
  = -\nabla_{R_A} E_2,\quad s_9 = 0.
\]

Hard gates: **dftd4 3.7.0** \(E_{\mathrm{disp}}\) + standalone forces
(geometry only, k-mesh independent); **CP2K 2026.1** multi-k FORCE Δ
(D4 − plain); multi-k **wiring** identity after one shared SCF; fair wall
ratio **2.426**.

Related: Γ two-body forces — [gpw-d4-forces](gpw-d4-forces.md); multi-k ATM —
[gpw-d4-atm-kpoint-forces](gpw-d4-atm-kpoint-forces.md); multi-k D4 stress —
[gpw-d4-kpoint-stress](gpw-d4-kpoint-stress.md); plain multi-k forces —
[gpw-kpoint-forces](gpw-kpoint-forces.md).
