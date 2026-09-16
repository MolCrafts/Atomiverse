# Reference — PE united-atom chain, bonded-only NVE energy conservation

## The physical system

A single all-trans C₂₀H₄₂ polyethylene **united-atom** chain (CH₂ beads) with
**bonded interactions only** (bond + angle + multi/harmonic dihedral). Non-bonded
LJ is intentionally off so the case isolates the integrator × bonded force
pipeline.

Bonded parameters follow the OPLS-AA UA / MSU PE deck used throughout the
Atomiverse PE path (Tschopp & McDowell, *J. Mech. Phys. Solids* **58**, 187–206
(2010) and the published MSU LAMMPS PE setup):

| Term | Form (LAMMPS convention) | Values |
|---|---|---|
| Bond | \(V = K(r-r_0)^2\) | \(K = 268\) kcal mol⁻¹ Å⁻², \(r_0 = 1.53\) Å |
| Angle | \(V = K(\theta-\theta_0)^2\) | \(K = 60\) kcal mol⁻¹ rad⁻², \(\theta_0 = 109.5°\) |
| Dihedral | \(V = \sum_{n=0}^{4} C_n \cos^n\phi\) | \(C = \{1.73,-4.49,0.776,6.99,0\}\) kcal mol⁻¹ |

Masses: 14.027 amu per CH₂ bead. Timestep \(\Delta t = 0.5\) fs (tight for
bonded high-frequency modes). A tiny alternating transverse velocity kick
(\(\pm 10^{-3}\) Å/fs) puts the chain on a non-trivial trajectory without
leaving the harmonic basin of each bonded term.

## What is asserted (not a golden energy)

Same theoretical statement as `md/lj_argon` — **Velocity Verlet is symplectic**:

> Total energy error is *bounded* and *oscillatory* — no secular drift.
> Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., §4.3.4;
> Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2;
> Hairer, Lubich & Wanner, *Geometric Numerical Integration*, Ch. IX.

Checks:

1. **Bounded.** \(\max |E(t)-E(0)| / |E(0)| \le 10^{-4}\) over 1000 steps
   (500 fs). Tighter than the LJ argon envelope because \(\Delta t\) is small
   and the force field is purely bonded (no neighbor-list skin).
2. **Non-secular.** Worst drift in the second half of the trajectory may not
   exceed 4× the worst drift in the first half (sampled in 20 chunks).

Absolute bonded energies at fixed geometry are pinned as **unit** tests in
`tests/cpu/md/test_pe_single_chain.cpp` and `test_pe_fd_force.cpp` against
closed-form analytics — not here.

## Why this case exists beside `lj_argon`

| Case | What the MD path must get right |
|---|---|
| `lj_argon` | pair LJ + neighbor list + min-image + VV |
| `pe_bonded_nve` | bond / angle / dihedral topology + VV (no pair list) |

Both are literature-grounded conservation properties of a symplectic
integrator, not freezes of our own energies.
