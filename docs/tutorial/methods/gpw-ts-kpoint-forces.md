# Multi-k GPW free-atom TS analytic ionic forces

*(Technical-report page; CP2K capability loop. Hard-coded offline pins only.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-ts-kpoint-forces` |
| Regression | `regressions/gpw/gpw-ts-kpoint-forces/` |
| Public API | `cpu::gpw::GpwDft::ts` + `k_mesh` / `GPWEvaluator::ts`; multi-k `gpw_forces_from_scf` / `GPWEvaluator(compute_forces=true)`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion + §10 Periodic KS |
| Oracle | **libMBD** free-atom TS energy + force dual-gate (CP2K has **no** stock `TYPE TS`) |

## Summary

**Monkhorst–Pack multi-k GPW** ionic forces with additive **Tkatchenko–Scheffler
(TS)** free-atom volumes. Γ already folds TS into `assemble_gpw_forces`
(`gpw-ts-forces`); multi-k stress is `gpw-ts-kpoint-stress`. This capability is
the **dedicated multi-k product force pin**: the same geometry-only

\[
F_{\mathrm{TS}} = -\nabla E_{\mathrm{disp}}^{\mathrm{TS}}
\]

is added on `assemble_gpw_forces_kpoints` when `ts` is set.

Product ionic force:

\[
\mathbf{F}_A^{\mathrm{tot}}
  = \mathbf{F}_A^{\mathrm{k\text{-}HF/Pulay}}
  + \mathbf{F}_A^{\mathrm{TS}},
\quad
\mathbf{F}_A^{\mathrm{TS}}
  = -\nabla_{R_A} E_{\mathrm{disp}}^{\mathrm{TS}}
  \quad (V/V_{\mathrm{free}}=1).
\]

Hard gates: offline **libMBD** \(E_{\mathrm{disp}}\) (water dimer free-atom) +
**force dual-gate** + multi-k **wiring** on compressed H₂. Wall ratio is
**perf_exempt** (no stock CP2K TS twin — same policy as `gpw-ts-forces`).

## Method parameters (PBE TS)

| Parameter | Value |
| --------- | ----: |
| \(s_R\) | 0.94 |
| Fermi \(d\) | 20 |
| Periodic pair cutoff | 50 Å (pure real-space lattice sum) |

Free-atom tables: `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).

## Public surface

```cpp
cpu::gpw::GpwDft gpw;
gpw.k_mesh[0] = 1; gpw.k_mesh[1] = 1; gpw.k_mesh[2] = 2;
gpw.ts = true;
gpw.build(atoms);
const auto& scf = gpw.run(atoms);
const auto F = cpu::gpw::gpw_forces_from_scf(atoms, gpw, scf);
// scf.e_disp [Ha] is geometry-only TS
```

No `*Config` bag; no `make_*` factory.

## Pins and geometry

See `regressions/gpw/gpw-ts-kpoint-forces/REFERENCE.md` and
`tools/cp2k_oracle/inputs/gpw-ts-kpoint-forces/h2_ts_kpoint_forces.note`.
