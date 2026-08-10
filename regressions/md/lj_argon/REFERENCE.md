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

## Devices — one case, multi-device

This is **one regression** with a PyTorch-style device switch
(`--device=cpu|cuda|both`, default `both`). When `ATV_BUILD_CUDA=ON` the
`.cu` TU replaces the `.cpp` target so one binary runs both backends;
otherwise the `.cpp` builds the CPU leg only. `oracle: literature` on both
legs (the gates below are analytic properties, not engine snapshots), plus a
same-settings CPU↔CUDA parity gate when both legs run.

| Leg | Type | Where the work runs |
| --- | ---- | ------------------- |
| `cpu` | `cpu::md::MD` | everything on the host |
| `cuda` | `cuda::md::MD` | the whole step loop on the GPU: OpenMM-style cluster/tile neighbor rebuild + LJ force, velocity-Verlet integration, energy/KE reductions. Host per chunk: the explicit D2H retrieve of `run()`. |

**Device-path honesty:** unlike `cuda::scf::RHF` (which can grade a host
fallback outside its device envelope), `cuda::md::MD` has **no host fallback**
— every component the CUDA leg registers (`cuda::PairLJCut`,
`cuda::NeighborList`, `cuda::md::VelocityVerlet`, `cuda::md::InitVelocity`) is
a device component whose `launch()` is a kernel, and the host frame is touched
only at the explicit H2D/D2H boundaries of `cuda::md::MD::run`. A green
`--device=cuda` leg therefore provably exercised the device MD path.

The two legs draw **different Maxwell velocities** (different RNGs at the same
seed), so they follow different micro-trajectories. That is irrelevant to
gates 1–2, which are properties of the integrator on *any* trajectory of this
Hamiltonian — and it is exactly why gate 3 compares a frozen lattice instead
of a trajectory.

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

Two independent checks follow from that one statement, applied **per device**:

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

## Gate 3 — same-settings CPU↔CUDA lattice-energy parity

With `--device=both` the binary additionally evaluates the shifted-LJ pair
energy of the **frozen simple-cubic lattice** on each backend and asserts

|E_cuda − E_cpu| / N ≤ 1e-9 eV/atom.

The perfect SC lattice is inversion-symmetric, so the net force on every atom
is exactly zero and one velocity-Verlet step from v = 0 moves nothing: the
energy read after that step is the lattice pair sum at **bit-identical
positions** on both backends. This isolates pair kernel + neighbor list +
minimum image from the backend-specific velocity RNG. Both work tiers default
to double; the only admissible difference is summation order/FMA contraction
(~1e-13 relative, ~1e-14 eV/atom here), so 1e-9 eV/atom is generous against
FP noise and seven orders of magnitude below the physics scale ε ≈ 1e-2 eV.
Design-time bound — re-pin from the measured GH200 gap before ever tightening.

Per the ES/CUDA oracle law the CPU leg is a **parity** reference, not a
literature number: it carries no independent physical authority. Absolute
correctness of the LJ forces against an external engine (LAMMPS) is **not**
pinned in-tree today — the old `benchmarks/md/lj/validation/` case was removed
with the benchmarks teardown; what this case pins is the analytic conservation
property (gates 1–2, each backend) and cross-backend consistency (gate 3).

## Why energy conservation is the right thing to pin here

This is the seed case of the `regressions/` harness. It exercises the whole public
MD path end-to-end — `Frame` → `PairLJCut` → `NeighborList` → `VelocityVerlet` →
`MD::run` — against a criterion that is analytic, parameter-free, and cannot be
satisfied by a subtly wrong force field: a wrong force can still conserve *its own*
energy, but it cannot conserve energy while the neighbor list, the min-image, and
the integrator disagree with each other. On the CUDA leg the same criterion
covers the OpenMM-style cluster/tile engine: a stale cluster mask, a lost tile
interaction, or a broken spatial reorder all leak energy over 10 ps.

## Measurement log

| Item | Value |
| ---- | ----- |
| GH200 job id | *pending — filled from the validation run* |
| Toolchain | `GPU/buildenv-gcccuda/2026.03-cu13.0` |
| Gate 1–2 (`cpu`) | *pending* |
| Gate 1–2 (`cuda`) | *pending* |
| Gate 3 (parity) | *pending* |

## Performance

No performance claim. This slug is a correctness/parity gate on a 216-atom
cell — small systems are correctness-only and are never the "CUDA wins" story
(ES optimization order, step 4; the same freeze applies to MD CUDA claims).
No wall-clock ratio is recorded and none may be inferred from it.
