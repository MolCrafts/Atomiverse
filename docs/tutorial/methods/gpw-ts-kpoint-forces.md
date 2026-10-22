# Multi-k GPW free-atom TS analytic ionic forces

*(Technical-report page; CP2K capability loop. Hard-coded offline pins only.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft::ts` + `k_mesh`; multi-k `GpwDft::forces(frame)` / `GPWEvaluator(std::move(dft), /*compute_forces=*/true)`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion + §10 Periodic KS |
| Oracle | **libMBD** free-atom TS energy + force dual-gate (CP2K has **no** stock `TYPE TS`) |

## Summary

**Monkhorst–Pack multi-k GPW** ionic forces with additive **Tkatchenko–Scheffler
(TS)** free-atom volumes. Γ already folds TS into `assemble_gpw_forces`
(`gpw-ts-forces`); multi-k stress is the matching `stress(frame)` path. This capability is
the **multi-k force path**: the same geometry-only

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
cpu::gpw::GpwDft gpw{kinds};   // one Kind per element
gpw.k_mesh[0] = 1; gpw.k_mesh[1] = 1; gpw.k_mesh[2] = 2;
gpw.ts = true;
gpw.build(frame);                     // positions [Å], Z in kAtom_type, box [Å]
const auto& scf = gpw.forces(frame);  // SCF + analytic forces → kAtom_f* [eV/Å]
// scf.e_disp [Ha] is geometry-only TS
```

No `*Config` bag; no `make_*` factory.

## Pins and geometry

See the pin notes recorded when this page was written (no standing regression) and
CP2K oracle.
