# sampling/umbrella

One case; every section below is run by `atv_regressions --case sampling/umbrella` and keeps its own oracle record.


## Reference — multi-window umbrella + WHAM smoke

### Physics

`U_bias = ½ k (ξ − ξ0)²` with `ξ = |r_i − r_j|` (Torrie & Valleau 1977). The
engine-side path is `HarmonicBias` + `DistanceCV` on Evaluate.

For multi-window reconstruction the regression draws exact Boltzmann samples
of each window bias (true PMF constant ⇒ Gaussian ξ ~ N(ξ0, 1/√k)) and runs a
fixed-point WHAM (Kumar et al. 1992) in-process. The recovered F(ξ) must be
flat with RMSE < 0.3 kBT after mean subtraction.

### Checks

1. Analytic single-window U and F on Evaluate (unit physics).
2. Six overlapping windows (Δξ = 0.4, k = 16) with on-centre energy ≈ 0 via
   public APIs.
3. WHAM flat-PMF RMSE gate.

### Ownership

- Atomiverse owns CV evaluation, bias forces, and sample production.
- WHAM equations and production PMF analysis belong in molpy; this regression
  carries a minimal WHAM only as a scientific smoke gate until molpy ships a
  WHAM solver.


## Reference — angle CV + harmonic umbrella

ξ = ∠ijk [rad]. At a right-angle geometry with window centre θ0 = π/2 and
bias k, energy and force vanish.

- Torrie & Valleau umbrella sampling
- Wilson B-matrix angle gradients
