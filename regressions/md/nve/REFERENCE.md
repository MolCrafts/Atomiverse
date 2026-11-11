# md/nve

One case; every section below is run by `atv_regressions --case md/nve` and keeps its own oracle record.


## Reference — LJ argon, NVE energy conservation

### The physical system

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

### Devices — one case, multi-device

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

### What is asserted, and why it is not a golden value

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

### Gate 3 — same-settings CPU↔CUDA lattice-energy parity

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
pinned by this case (see `benchmarks/compare/lj-lammps` for the LAMMPS
comparison); what this case pins is the analytic conservation
property (gates 1–2, each backend) and cross-backend consistency (gate 3).

### Why energy conservation is the right thing to pin here

This is the seed case of the `regressions/` harness. It exercises the whole public
MD path end-to-end — `Frame` → `PairLJCut` → `NeighborList` → `VelocityVerlet` →
`MD::run` — against a criterion that is analytic, parameter-free, and cannot be
satisfied by a subtly wrong force field: a wrong force can still conserve *its own*
energy, but it cannot conserve energy while the neighbor list, the min-image, and
the integrator disagree with each other. On the CUDA leg the same criterion
covers the OpenMM-style cluster/tile engine: a stale cluster mask, a lost tile
interaction, or a broken spatial reorder all leak energy over 10 ps.

### Measurement log

| Item | Value |
| ---- | ----- |
| GH200 job id | *pending — filled from the validation run* |
| Toolchain | `GPU/buildenv-gcccuda/2026.03-cu13.0` |
| Gate 1–2 (`cpu`) | *pending* |
| Gate 1–2 (`cuda`) | *pending* |
| Gate 3 (parity) | *pending* |

### Performance

No performance claim. This slug is a correctness/parity gate on a 216-atom
cell — small systems are correctness-only and are never the "CUDA wins" story
(ES optimization order, step 4; the same freeze applies to MD CUDA claims).
No wall-clock ratio is recorded and none may be inferred from it.


## Reference — PE united-atom chain, bonded-only NVE energy conservation

### The physical system

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

### What is asserted (not a golden energy)

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

### Why this case exists beside `lj_argon`

| Case | What the MD path must get right |
|---|---|
| `lj_argon` | pair LJ + neighbor list + min-image + VV |
| `pe_bonded_nve` | bond / angle / dihedral topology + VV (no pair list) |

Both are literature-grounded conservation properties of a symplectic
integrator, not freezes of our own energies.


## Initial-force CPU Velocity Verlet trajectory

oracle: literature

W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson,
“A computer simulation method for the calculation of equilibrium constants
for the formation of physical clusters of molecules: Application to small
water clusters,” *Journal of Chemical Physics* **76**, 637–649 (1982),
[doi:10.1063/1.442716](https://doi.org/10.1063/1.442716).

The velocity Verlet update uses the force at the original coordinates for
its first half-kick:

    x(t+dt) = x(t) + dt*v(t) + dt²*a(x(t))/2
    v(t+dt) = v(t) + dt*[a(x(t)) + a(x(t+dt))]/2.

Two atoms of masses 2 and 3 amu start at x = 4 and 6.4 Å with zero full-step
velocities. One harmonic bond has K = 7 eV/Å² and equilibrium length 1.2 Å,
with V = K(r−r0)². The public unit conversion supplies a = F/m in Å/fs².
The regression compares positions and velocities after one 0.5 fs step and
a second continued step with these analytic updates. It independently runs
two steps in one call and verifies that chunking does not restart the initial
force pass or lose the current force. Transverse coordinates and velocities
remain unchanged. No force component lifecycle is driven by the consumer.

`run(frame, 0)` must overwrite deliberately stale input forces with the
analytic initial force (±16.8 eV/Å) and energy (10.08 eV), without moving
positions or velocities. Maximum absolute trajectory error is below 1e−11.


## Initial-force CUDA Velocity Verlet trajectory

oracle: literature

W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson,
“A computer simulation method for the calculation of equilibrium constants
for the formation of physical clusters of molecules: Application to small
water clusters,” *Journal of Chemical Physics* **76**, 637–649 (1982),
[doi:10.1063/1.442716](https://doi.org/10.1063/1.442716).

The velocity Verlet update uses the force at the original coordinates for
its first half-kick:

    x(t+dt) = x(t) + dt*v(t) + dt²*a(x(t))/2
    v(t+dt) = v(t) + dt*[a(x(t)) + a(x(t+dt))]/2.

Two atoms of masses 2 and 3 amu start at x = 4 and 6.4 Å with zero full-step
velocities. One harmonic bond has K = 7 eV/Å² and equilibrium length 1.2 Å,
with V = K(r−r0)². The public unit conversion supplies a = F/m in Å/fs².
The regression compares positions and velocities after one 0.5 fs step and
a second continued step with these analytic updates. It independently runs
two steps in one call and verifies that chunking does not restart the initial
force pass or lose the current force. Transverse coordinates and velocities
remain unchanged. No force component lifecycle is driven by the consumer.

Maximum absolute position/velocity error must be below 1e−11 in double or
3e−6 when force math (`force_t`) or integration state (`mixed_t`) is FP32. Four total integration steps on two atoms should
finish within ten seconds on Arrhenius, including CUDA initialization.


## Multi-GPU velocity-Verlet trajectory and topology transport

This public-library consumer checks that domain decomposition preserves the
single-GPU velocity-Verlet trajectory and transports the complete bonded
topology when atoms migrate. It requires at least two visible CUDA GPUs and
returns failure if that hardware is unavailable.

The integration reference is W. C. Swope, H. C. Andersen, P. H. Berens and
K. R. Wilson, *A computer simulation method for the calculation of equilibrium
constants for the formation of physical clusters of molecules: Application to
small water clusters*, J. Chem. Phys. **76**, 637–649 (1982),
[doi:10.1063/1.442716](https://doi.org/10.1063/1.442716).
The decomposition invariant is the same physical trajectory under a partition
of the atoms, as described by S. Plimpton, *Fast Parallel Algorithms for
Short-Range Molecular Dynamics*, J. Comput. Phys. **117**, 1–19 (1995),
[doi:10.1006/jcph.1995.1039](https://doi.org/10.1006/jcph.1995.1039).

The four cases retain the original fixtures, integration steps and assertions:

| Case | Atoms | Steps | Position tolerance |
|---|---:|---:|---:|
| `NveTrajectoryParity` | 64 | 10 | 0.005 Å per component |
| `NveTrajectoryParityNlist` | 512 | 12 | 0.005 Å per component |
| `NveMigrationParity` | 512 | 20 | 0.02 Å per component |
| `MigratePreservesBondedTopology` | 64 | 20 | Exact equality of bond-tag sets |

Run `reg_cuda-multi-gpu-trajectory CASE` to check one case, or omit `CASE` to
run all four. The migration fixture uses the original +x drift and requires
the complete topology after every-step migration; no tolerance is applied to
the bond-tag set. The position checks use the serial GPU as their numerical
reference, rather than a frozen trajectory from another engine.
