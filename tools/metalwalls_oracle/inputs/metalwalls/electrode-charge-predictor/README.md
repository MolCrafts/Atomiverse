# Offline oracle notes — electrode-charge-predictor

oracle: metalwalls  
Feature: Kolafa ASPC electrode charge warm-start for CG/BO (MetalWalls E2).

LAMMPS ELECTRODE has no ASPC path → algorithm pin from Kolafa (2004) +
MetalWalls const-pot CG documentation (Coretti et al. JCP 2022).

Consumed by: `regressions/md/electrode-charge-predictor/REFERENCE.md`.
No live MetalWalls binary in ctest.
