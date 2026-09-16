# Reference — ForceField bag → Evaluate single-point

## Roles

- `cpu::ForceField` — owns potential objects (`set_potential` / `set_evaluator` / …).
- `cpu::Evaluate` — one-pass **Driver** preset (single-point energy/force).
- `ff.apply(eval)` moves the bag into the driver before `build`/`run`.

## Checks

Two waters with LJ + coul/cut + harmonic bond/angle and `special_bonds 0 0 0`.
Energy finite, |E| < 50 eV.
