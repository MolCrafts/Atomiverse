---
title: NDDO RM1 CO2/CO discriminator
status: in-progress
created: 2026-08-25
slug: nddo-sharp-discriminator
oracle: mopac
---

# nddo-sharp-discriminator

## Summary

Water is a mild NDDO gate: RM1 ΔHf(H₂O) sits 0.06 kcal/mol from experiment, so
a passing water pin does not prove the *method*. Pin RM1 heats of formation
for CO₂ and CO against the OpenMOPAC PARAM survey — the same source as
`reg_rm1-water-mopac-energy` — where RM1 is **12.5 kcal/mol** from experiment.

No new public type. Existing `cpu::semiempirical::RM1`. Frame in / Frame out.

## Domain basis

RM1 (Rocha, Freire, Simas, Stewart, *J. Comput. Chem.* **27**, 1101 (2006)).
MOPAC-style heat of formation:

\[
\Delta H_f = (E_{\mathrm{elec}}+E_{\mathrm{core}} - \sum_A E_{\mathrm{isol},A})
\cdot (eV\to kcal/mol) + \sum_A \Delta H_{f,\mathrm{exp}}(A)
\]

A published MOPAC ΔHf belongs to the **method's own** stationary geometry
(OpenMOPAC `data/README.md`). Relax the unique CO₂ / CO bond on `RM1::run`.

## Pin (not from ATV)

OpenMOPAC PARAM survey `Analysis/Survey organic/RM1.heats` @ `9859905`
(Apache-2.0), the same file the water pin cites:

| Species | Exp. | RM1 calc. | Diff. | `RM1.geos` r(C–O) calc. |
| ------- | ---- | --------- | ----- | ----------------------- |
| CO₂     | −94.05 | **−81.55** | 12.50 | 1.173 Å |
| CO      | −26.42 | **−13.87** | 12.55 | 1.138 Å |

Secondary published table of *errors* (not used as the pin, cited for
cross-check): [openmopac.net accuracy table](https://openmopac.net/PM7_and_PM6-D3H4_accuracy/table_of_heats.html)
RM1 error on CO₂ = +12.50 kcal/mol on ΔHf_exp = −94.05.

**Family lock:** OpenMOPAC only. Never DFTB+.

## Design

- New regression `reg_nddo-sharp-discriminator` only.
- Public API unchanged: `RM1{}.build(frame); rm1.run(frame)`.
- No `*Config`, no factory, no `set_region`.
- Twin: `tools/tb_oracle/inputs/mopac/rm1-co2/` (reconstructed decks; pin
  numbers from PARAM survey, not from a live mopac run on this host).

## Acceptance

See [nddo-sharp-discriminator.acceptance.md](nddo-sharp-discriminator.acceptance.md).

## Files

- `regressions/semiempirical/nddo-sharp-discriminator/`
- `tools/tb_oracle/inputs/mopac/rm1-co2/`
- `regressions/CMakeLists.txt` (register the case)
