# nddo-sharp-discriminator — RM1 CO₂ / CO vs OpenMOPAC PARAM survey

```text
oracle: mopac
oracle_version: OpenMOPAC PARAM survey tables (github.com/openmopac/param @ 9859905,
                Apache-2.0); cross-checked against openmopac.net PM7/PM6-D3H4
                accuracy table (RM1 error column, 2021-03-11 dump)
family_oracle: mopac
```

NDDO family lock: **OpenMOPAC only**. No DFTB+ number appears in this case.
No `mopac` binary exists on the Atomiverse build host and ctest never shells
out to one.

## Why this case exists

`reg_rm1-water-mopac-energy` pins RM1 ΔHf(H₂O) = −57.86 kcal/mol against
experiment −57.80 (0.06 kcal/mol). That gate can pass if the code merely
reproduces *thermochemistry of water*, not RM1. Carbon dioxide and carbon
monoxide are the survey rows where RM1 is **12.5 kcal/mol** from experiment.

## Method

**RM1** — G. B. Rocha, R. O. Freire, A. M. Simas and J. J. P. Stewart,
*J. Comput. Chem.* **27**, 1101–1111 (2006). Same public shape as the water
pin: `cpu::semiempirical::RM1`, members, `run(frame)`.

A MOPAC ΔHf belongs to the **method's own** stationary geometry (OpenMOPAC
`data/README.md`). This case relaxes the unique C–O distance on the RM1
surface (linear CO₂ / diatomic CO) with a 1-D pattern search over `RM1::run`.

## Pin

OpenMOPAC PARAM survey `Analysis/Survey organic/RM1.heats` @ `9859905`:

```text
 Formula        Molecule                              Heat of Formation   Diff. Ref.
                                                       Exp.     Calc.
 CO            Carbon monoxide                       -26.42    -13.87    12.55     5
 CO2           Carbon dioxide                        -94.05    -81.55    12.50     5
```

`RM1.geos` (experimental column first, RM1-calculated second):

```text
 CO            Carbon monoxide     C-O    1.128   1.138
 CO2           Carbon dioxide      C-O    1.162   1.173
```

| Quantity | Published RM1 | Source | Tol |
| -------- | ------------- | ------ | --- |
| ΔHf(CO₂, relaxed) | **−81.55** kcal/mol | `RM1.heats` L829 | ±0.05 kcal/mol |
| r(C–O) CO₂ | **1.173** Å | `RM1.geos` L225 | ±0.005 Å |
| ΔHf(CO, relaxed) | **−13.87** kcal/mol | `RM1.heats` L548 | ±0.05 kcal/mol |
| r(C–O) CO | **1.138** Å | `RM1.geos` L167 | ±0.005 Å |

**Discriminator (binding):** `|ΔHf_RM1(CO₂) − (−94.05)| > 10 kcal/mol`. If a
future RM1 table accidentally tracked experiment, this case must fail — the
point is to pin the *method*, not JANAF.

Cross-check (errors, not the pin): openmopac.net accuracy table, CO₂ row,
RM1 error = +12.50 kcal/mol on ΔHf_exp = −94.05 → calc = −81.55.

## Twin

`tools/tb_oracle/inputs/mopac/rm1-co2/` — reconstructed `RM1` decks at the
experimental start geometry. The pin numbers are the PARAM survey, not a
transcript of those decks on this host.

## Tolerance rationale

PARAM survey prints two decimals (0.01 kcal/mol). The NDDO constant-convention
floor vs a live MOPAC run is ~0.02 kcal/mol (see `am1-water-mopac-energy`).
This case pins a *published survey table*, not a MOPAC 7.00 energy in eV, so
±0.05 kcal/mol is the gate (five times the print precision, twice the
convention floor). Geometry: survey prints 0.001 Å; ±0.005 Å covers the 1-D
search termination.

## Gaps

No live OpenMOPAC binary, so there is no ATV↔mopac wall. `perf_exempt`.
No closed-form NDDO forces here (see `nddo-closed-form-forces`).
