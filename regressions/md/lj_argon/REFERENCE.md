# Reference — LJ argon, NVE energy conservation

## The physical system

A. Rahman, **"Correlations in the Motion of Atoms in Liquid Argon"**,
*Phys. Rev.* **136**, A405 (1964). DOI 10.1103/PhysRev.136.A405 — the paper that
founded molecular dynamics of a realistic liquid.

Its model, used verbatim here:

| Quantity | Rahman's value | In this case |
|---|---|---|
| Potential | Lennard-Jones 12-6 | `PairLJCut` |
| ε/k_B | 120 K | `eps = 120 * kB` = 0.010341 eV |
| σ | 3.4 Å | `sigma = 3.4` |
| Number density | 1.374 g cm⁻³ → **0.02071 Å⁻³** | `number_density = 0.02071` |
| Temperature | 94.4 K | `temperature = 94.4` |
| Timestep | 10⁻¹⁴ s = 10 fs | **5 fs** (deliberately tighter; see below) |
| N | 864 | **216** (regression must be ≤ 60 s) |

Cutoff is 2.5σ = 8.5 Å, the standard LJ truncation (Frenkel & Smit,
*Understanding Molecular Simulation*, 2nd ed., §3.2.2 and Table 4.1).

We reduce N from 864 to 216 and the timestep from 10 fs to 5 fs. Neither changes
what is being tested: energy conservation is a property of the **integrator**, not
of the system size, and a smaller dt only makes the criterion below stricter.

## What is asserted, and why it is not a golden value

The assertion is **not** "the total energy equals some number we recorded". It is
the qualitative property that a symplectic integrator must have, taken from theory:

> **Velocity Verlet is symplectic.** Its energy error is therefore *bounded* and
> *oscillatory* — it does not accumulate. There is no secular (linear-in-time)
> drift, and the size of the bounded oscillation is O(dt²).
>
> — Frenkel & Smit §4.3.4 ("the total energy … should be conserved"); Allen &
> Tildesley, *Computer Simulation of Liquids*, 2nd ed., §3.2; Hairer, Lubich &
> Wanner, *Geometric Numerical Integration*, Ch. IX (backward error analysis: the
> symplectic map conserves a shadow Hamiltonian, so H stays within O(dt²) of it
> for exponentially long times).

Two independent checks follow from that one statement:

1. **Bounded.** `max_t |E(t) − E(0)| / (N·ε) ≤ 1e-2`.
   The tolerance is an *envelope*, not a measurement: it says the error stays
   within one percent of ε per atom, i.e. small against the depth of the potential
   well that sets the energy scale of the whole problem. It is deliberately loose —
   the point is to catch a broken integrator, a broken neighbor list, or a broken
   min-image (each of which produces drift orders of magnitude larger), not to
   pin a digit. Tightening it to fit whatever we currently measure would convert
   this case into a golden-value test of our own bugs, which the `regressions/`
   rules forbid.

2. **Non-secular.** The worst drift in the second half of the trajectory may not
   exceed 4× the worst drift in the first half.
   This is the check with teeth. A bounded-but-oscillating error passes; an energy
   *leak* — the classic signature of a neighbor-list skin that is too small, of a
   missed force term, or of a non-time-reversible integrator — ramps with time and
   fails, even if its magnitude at the final step happened to land inside the
   envelope. Endpoint-only conservation checks miss exactly this, which is why the
   run is sampled in 20 chunks.

## Why energy conservation is the right thing to pin here

This is the seed case of the `regressions/` harness. It exercises the whole public
MD path end-to-end — `Frame` → `PairLJCut` → `NeighborList` → `VelocityVerlet` →
`MD::run` — against a criterion that is analytic, parameter-free, and cannot be
satisfied by a subtly wrong force field: a wrong force can still conserve *its own*
energy, but it cannot conserve energy while the neighbor list, the min-image, and
the integrator disagree with each other.

Absolute correctness of the LJ *forces* is pinned separately, against LAMMPS, in
`benchmarks/md/lj/validation/` — that is an external-engine parity check and
belongs there, not here.
