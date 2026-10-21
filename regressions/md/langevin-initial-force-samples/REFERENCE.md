# Independent noise in the initial Langevin half-kicks

oracle: literature

References:

- A. Einstein, “Über die von der molekularkinetischen Theorie der Wärme
  geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen,”
  *Annalen der Physik* **322**, 549–560 (1905),
  [doi:10.1002/andp.19053220806](https://doi.org/10.1002/andp.19053220806).
- T. Schneider and E. Stoll, “Molecular-dynamics study of a three-dimensional
  one-component model for distortive phase transitions,” *Physical Review B*
  **17**, 1302–1322 (1978),
  [doi:10.1103/PhysRevB.17.1302](https://doi.org/10.1103/PhysRevB.17.1302).

The pin derives from the Langevin fluctuation-dissipation relation: additive
white-noise force has strength proportional to `2 m kB T / damp`, and random
vectors at distinct force evaluations are independent. The following finite-step
moment is an analytic consequence of this relation and the two velocity-Verlet
half-kicks; it is not a number measured from Atomiverse or another engine.

There are 16,384 independent, noninteracting atoms, each of mass 12 amu,
initially at rest in a 200 Å cubic box. Tags are 0 through 16,383. Both
backends use temperature 300 K, damping time 100 fs, timestep 0.5 fs and
seed 173. No COM subtraction, pair evaluator, neighbor list, potential force
or velocity initializer is mounted. The only force comes from `FixLangevin`.

Let `M = units::mass_v2_to_energy`, the conversion from amu·(Å/fs)² to eV.
For one velocity component, define

    c² = dt kB T / (2 m M damp),       a = dt / (2 damp).

The initial force evaluation draws a standard Gaussian `ξ0`, giving the first
half-kick `v_half = c ξ0`. At the updated position the thermostat adds drag
from this half-step velocity and a new, independent standard Gaussian `ξ1`.
After exactly one integration step,

    v1 = c [(1-a) ξ0 + ξ1],
    E[v1²] = c² [(1-a)² + 1].

The executable uses only public `MD`, `VelocityVerlet`, `FixLangevin` and
`Frame` composition. Each backend runs exactly one step, and the observed
second moment pools all 49,152 velocity components. CPU always runs; a CUDA
build also runs the CUDA case independently. Individual random trajectories
need not match across backends.

The ratio of observed to analytic second moment must be in **[0.96, 1.04]**
on each backend. For independent Gaussian components the relative standard
deviation of this statistic is `sqrt(2/49152) ≈ 0.00638`; the fixed 4% window
is approximately 6.3 standard deviations. Reusing the same Gaussian vector
for both initial half-kicks instead produces ratio

    (2-a)² / [(1-a)² + 1] ≈ 2,

which fails this unchanged window. Omitting the initial force gives a ratio
near one half and also fails. This case checks initial noise independence
and its fluctuation-dissipation amplitude; it makes no statement about
long-time equilibration, finite-step equilibrium bias, or transport.

Run the standalone regression with `OMP_NUM_THREADS=1` on Arrhenius. One CPU
step plus one CUDA step is intended to finish within ten seconds, including
CUDA initialization; no performance comparison is claimed.
