# nddo-sharp-discriminator — acceptance

```text
acceptance_met: true
```

Flip to `true` only after dual gate green on this worktree.

## Gates

1. **Pin, not thermochemistry.** At the RM1-relaxed linear CO₂ geometry,
   `|ΔHf_ATV − (−81.55 kcal/mol)| ≤ 0.05`. Experiment −94.05 must lie
   **outside** that window (discriminator: `|RM1 − exp| > 10 kcal/mol`).
2. **Geometry the heat belongs to.** Relaxed r(C–O) matches PARAM `RM1.geos`
   1.173 Å within 0.005 Å. Forces at that point `|F| < 0.05 eV/Å`.
3. **Second species.** RM1-relaxed CO: ΔHf within 0.05 kcal/mol of −13.87
   and r(C–O) within 0.005 Å of 1.138 Å.
4. **Oracle hygiene.** `REFERENCE.md` states `oracle: mopac` and
   `oracle_version`. Twin decks live under `tools/tb_oracle/inputs/mopac/rm1-co2/`.
   ctest never shells out to mopac.
5. **Architecture.** No public `*Config`, no `make_*`/`build_*`, Frame is the
   only I/O, SE-only cmake.

## Dual gate

- Correctness: the pins above.
- Perf: `perf_exempt` — three-atom SCF vs a published table; no mopac binary
  on this host to ratio against.
