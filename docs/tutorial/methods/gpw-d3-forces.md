# GPW PBE-D3 analytic ionic forces — isolated water dimer

*(Technical-report page; CP2K capability loop wave 14.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d3-forces` |
| Regression | `regressions/gpw/gpw-d3-forces/` |
| Public API | `cpu::gpw::GPWEvaluator{d3 = true, compute_forces = true, nonperiodic = true}` |
| Roadmap / queue | `docs/roadmap/dft.md` §8 Dispersion; queue done 2026-07-26 |

## Summary

Grimme **DFT-D3 with zero ("original") damping** contributes to the **ionic
forces** on the GPW path, as the exact gradient of the same additive
\(E_{\mathrm{disp}}\) the SCF converged — including the coordination-number
dependence of \(C_6\), which makes the gradient non-trivial. Twinned to two
offline **CP2K 2026.1** `ENERGY_FORCE` runs on an isolated water dimer.

## Why the pin is a difference of two CP2K runs

DFT-D3 is geometry-only: it never enters the Kohn–Sham potential, so the
converged density is identical with and without it. Therefore

\[
F_A(\mathrm{D3}) - F_A(\mathrm{plain}) = -\nabla_A E_{\mathrm{disp}},
\]

which isolates exactly the new capability while the much larger electronic force
cancels. Running the same geometry twice in CP2K — once with `&VDW_POTENTIAL`,
once without — gives that pin directly.

## Measured (2026-07-26)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1e−8 Ha | **6.4e−11** |
| \(F_{\mathrm{disp}}\) vs CP2K difference | 5e−6 Ha/Bohr | **7.8e−07** |
| Total \(F\) vs CP2K | 1e−2 Ha/Bohr | **7.5e−03** |
| \(\lvert\sum_A F_A\rvert\) | 1e−3 Ha/Bohr | **2.2e−04** |
| Wiring \((F_{d3}-F_{plain})\) vs standalone | 1e−7 Ha/Bohr | **9.1e−11** |
| Wall | ≤ 60 s | **17.8 s** (CP2K 12.3 s → ratio 1.4) |

Unit dual-gate: `GpwForces.AnalyticMatchesFdWithD3` — analytic vs multi-SCF
energy FD, **2.4e−6 Ha/Bohr** on cheap MT H₂.

## Two defects this gate exposed

1. **Pair-cutoff convention.** The zero-damping path hard-coded 40 Bohr in five
   sites while D3(BJ) used CP2K's documented `2 × R_CUTOFF`. A dedicated periodic
   D3-zero CP2K run settled it: 75.589 Bohr reproduces CP2K's `E_disp` to
   6.7e−11 Ha, 40 Bohr misses by 3.9e−7. All five sites now share
   `kD3Cp2kPairCutoffBohr`, so energy, force and stress differentiate the same
   functional.
2. **Martyna–Tuckerman placement.** With `nonperiodic = true` the collocation
   does not wrap, so a Gaussian near a cell face is clipped: the energy stays
   plausible while the forces gain a boundary term. At CP2K's origin-based
   coordinates `max|F_atv − F_cp2k|` was **31.9 Ha/Bohr**, falling to 7.5e−3 once
   the dimer is centred. `GpwDft::build` now throws on off-centre nonperiodic
   systems, and this regression asserts \(\lvert\sum F\rvert\) so the class is
   gated rather than documented.

## Performance

CP2K `ENERGY_FORCE` 12.3 s vs Atomiverse 17.8 s on the same host → ratio
**1.4**, inside the 10× fair wall. The Atomiverse figure includes one extra plain
SCF that only the wiring gate needs.

See `regressions/gpw/gpw-d3-forces/REFERENCE.md` for the full pin tables and
tolerance justifications.
