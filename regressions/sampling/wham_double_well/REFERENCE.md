# wham_double_well

## Physics

1D multi-window WHAM (Kumar et al., *J. Comput. Chem.* **13**, 1011 (1992)) on
synthetic histograms from the double-well
\(F(s)=\frac14(s^2-1)^2\) eV plus harmonic umbrellas. Solver lives in **molrs**;
Atomiverse exposes `cpu::compute::Wham`.

## Gate

Converged WHAM; barrier \(F(0)\) above wells \(F(\pm1)\) by ≥0.08 eV; wells
symmetric within 0.12 eV; reconstructed barrier height within 0.12 eV of 0.25 eV.
