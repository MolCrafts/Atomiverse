# Reference — AM1 water BOMD smoke

## Method

AM1 (Dewar et al., *J. Am. Chem. Soc.* **107**, 3902 (1985)) as the force
source of Born-Oppenheimer molecular dynamics: one AM1 SCF solve per step, the
nuclei integrated on the forces it publishes.

## Check

`cpu::md::MD` + `VelocityVerlet` + `AM1Evaluator` (gradients on) bound at
`md::Phase::force`, 20 steps at `dt = 0.25` fs (≈ 5 fs total), started from
`InitVelocity(T0 = 10 K, seed = 7)`. Tighter-than-default SCF settings —
`max_iter = 300`, `density_tol = 1e-7`, `energy_tol = 1e-9` — because a loose
SCF is what makes a short BOMD smoke flaky, not the integrator.

Gates, on the near-experimental water geometry the case builds:

- no throw (an unconverged SCF raises)
- potential energy `frame.meta[kEnergy]` finite and negative (bound molecule)
- kinetic energy `frame.meta[kKinetic]` finite and non-negative
- no energy explosion: `|PE| < 1e4` eV and `KE < 10` eV

Workflow pin only — not NVE conservation or a literature trajectory.
