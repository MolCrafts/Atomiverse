# H2 full-grid ALDA TDA

## System

H₂ at r = 0.74 Å, STO-3G, RKS-LSDA, closed-shell TDA singlet, full
MO-product ALDA \(f^{\mathrm{xc}}\) on the SG0 (50×110) grid.

## Oracle

Atomiverse `RKS::tda` full-grid ALDA path (Dirac-exchange second derivative),
2026-07-23. Not a third-party live oracle — value is hard-coded in the
regression binary after calibration against this engine.

## Gate

- `es.alda == true`
- S1 excitation energy [eV] within documented absolute tolerance
