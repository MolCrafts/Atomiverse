# OpenMOPAC offline twin — RM1 / CO₂ / CO

Twin for `regressions/semiempirical/nddo-sharp-discriminator`.

> **Offline only. ctest never executes these decks.** No `mopac` binary exists
> on the Atomiverse build host. Pin numbers are the OpenMOPAC PARAM survey,
> not a run of these decks here.

- **family:** NDDO → **OpenMOPAC only**
- **method keyword:** `RM1`

## Decks

Reconstructed start geometries (experimental r, linear). A regeneration host
with OpenMOPAC should run `RM1 PRECISE GNORM=0.001 XYZ` and compare
`FINAL HEAT OF FORMATION` to PARAM `RM1.heats`.

| File | Start geometry |
| ---- | -------------- |
| `rm1-co2.mop` | r(C–O) = 1.162 Å, linear |
| `rm1-co.mop` | r(C–O) = 1.128 Å |

## Expected (PARAM survey @ 9859905, not these decks)

| Species | RM1 ΔHf | RM1 r(C–O) |
| ------- | ------- | ---------- |
| CO₂ | −81.55 kcal/mol | 1.173 Å |
| CO | −13.87 kcal/mol | 1.138 Å |
