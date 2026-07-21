# Reference — multi-window umbrella + WHAM smoke

## Physics

`U_bias = ½ k (ξ − ξ0)²` with `ξ = |r_i − r_j|` (Torrie & Valleau 1977). The
engine-side path is `HarmonicBias` + `DistanceCV` on Evaluate.

For multi-window reconstruction the regression draws exact Boltzmann samples
of each window bias (true PMF constant ⇒ Gaussian ξ ~ N(ξ0, 1/√k)) and runs a
fixed-point WHAM (Kumar et al. 1992) in-process. The recovered F(ξ) must be
flat with RMSE < 0.3 kBT after mean subtraction.

## Checks

1. Analytic single-window U and F on Evaluate (unit physics).
2. Six overlapping windows (Δξ = 0.4, k = 16) with on-centre energy ≈ 0 via
   public APIs.
3. WHAM flat-PMF RMSE gate.

## Ownership

- Atomiverse owns CV evaluation, bias forces, and sample production.
- WHAM equations and production PMF analysis are intended to live in molrs;
  this regression carries a minimal WHAM only as a scientific smoke gate until
  the molrs streaming solver is pinned.
