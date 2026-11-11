# md/domain-decomposition

One case; every section below is run by `atv_regressions --case md/domain-decomposition` and keeps its own oracle record.


## Reference — swap-list halo == the single-image min-image sum (MPI DD)

oracle: literature
oracle_version: n/a (analytic identity; no oracle binary is run — the O(N²)
minimum-image sum is recomputed inside the regression. See citations below.)
perf_exempt: yes — this slug makes **no** performance claim. The swap-list
refactor replaces per-step, per-face geometry re-derivation with a table
derived once per rebuild; quantifying that is a separate slug
(`comm-topology-03`), and no wall-clock number is asserted here.

### Invariant under test

A domain decomposition is a **partition of one and the same sum**. Splitting a
periodic box across ranks may not change a single physical number: the halo
must hand every rank exactly the ghost copies its owned atoms need (forward,
`borders` + in-step `forward_comm`), and every partial force accumulated on a
ghost row must return to its owner **exactly once** (reverse, `post_force`).

Hence the criterion is an identity, not a remembered number:

| Quantity | Must equal |
| --- | --- |
| total potential energy after N steps | the flat O(N²) minimum-image sum over the same final configuration |
| every per-atom force | the same sum's analytic gradient |
| Σᵢ Fᵢ | 0 — non-zero means the reverse exchange ran zero times (force lost) or twice (energy pumped) |

The last row is the one that isolates reverse communication: a forward-only
bug shifts energy *and* forces, while a double-applied reverse leaves the
energy plausible and leaks momentum.

- S. Plimpton, *Fast Parallel Algorithms for Short-Range Molecular Dynamics*,
  J. Comput. Phys. **117**, 1–19 (1995), DOI 10.1006/jcph.1995.1039 — the
  original swap-list forward/reverse communication (`maxneed`, `sendlist`,
  `firstrecv`, `sendself`) whose shape the CPU halo now has.
- A. P. Thompson et al., *LAMMPS — a flexible simulation tool for
  particle-based materials modeling at the atomic, meso, and continuum
  scales*, Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171 — modern LAMMPS `borders` / `forward_comm` /
  `reverse_comm` semantics, including the "reverse exactly once" contract.
- A. Rahman, *Correlations in the Motion of Atoms in Liquid Argon*,
  Phys. Rev. **136**, A405 (1964), DOI 10.1103/PhysRev.136.A405 — the LJ
  argon parameters used here (ε/k_B = 120 K, σ = 3.4 Å, r_c = 2.5 σ).

### System (child, np = 2 and np = 1)

512-atom simple-cubic argon lattice, cubic box 28.936 Å, **forced 2×1×1
decomposition along x** (`ATV_FORCE_DIMS=2,1,1`; the override is ignored at
np = 1 because the product must equal the world size).

| Parameter | Value |
| --- | --- |
| Sites | 8³ = 512, at `(i + 0.5) · a₀` so none sits on the x = L/2 cut |
| a₀ | 3.617 Å (Rahman number density 0.02071 Å⁻³) |
| Box | 28.936 Å cubic, periodic |
| ε | 0.010341 eV (= 120 k_B) |
| σ | 3.4 Å |
| r_c | 8.5 Å (= 2.5 σ) |
| Verlet skin / reneighbor cadence | 0.85 Å / every 10 steps |
| Mass | 39.95 amu |
| dt / steps | 1.0 fs / 21 |
| Symmetry break | site (3, 3, 3) shifted 0.35 Å along (1,1,1)/√3 |

Pipeline — all defaults, the halo being the feature under test:
`PairLJCut{ε, σ, 8.5}` + `NeighborList{8.5, 0.85, every = 10}` +
`VelocityVerlet{1.0 fs}`. Halo / exchange / reverse live on `ctx.md.comm`
sized from `NeighborList::neighbor_reach()` (cutoff + skin = 9.35 Å).
Nothing here registers a halo component by hand.

#### Why this geometry

With a 14.468 Å subdomain against a 9.35 Å shell:

- `need_x = ceil(9.35 / 14.468) = 1` — the single-hop swap table this spec
  delivers (`nswap = 2 · Σ_d maxneed_d = 6`);
- `2 × 9.35 = 18.7 > 14.468` — the low-face and high-face bands **overlap**, so
  the same owner reaches its single Cartesian partner twice, as two distinct
  periodic images. That dual-image path is exactly what a "tidy-up" tag dedup,
  or a LAMMPS-style local memcpy on a `sendself` swap, would silently break;
- y and z are undecomposed (`dims = 1`) ⇒ both their swaps are `sendself`,
  contribute zero ghosts, and their periodicity is carried by the pair
  kernel's minimum image. `Σ nrecv` is therefore purely the x shell.

21 steps at the every = 10 cadence cross **3** full rebuilds (`borders`), with
in-step forward refreshes in between, so both forward paths are covered.

The oracle is computed **in this file**: an out-of-library `Component` at
`end_of_step @ +900` (after `mpi::Reduce` at −50 has published the globally
summed energy) allgathers every rank's owned rows by tag, reassembles the
global configuration, and evaluates the O(N²) minimum-image LJ sum and its
gradient. No third-party engine is imported, spawned, or read at build or run
time; no engine snapshot is stored.

### Pinned numbers (hard-coded gates in the regression)

Measured 2026-08-25 on the x86_64 login node (GCC + MPICH 4.3.2, Release,
`OMP_NUM_THREADS=1`); wall 0.48 s (budget ≤ 60 s).

| Gate | Want | Tolerance | Measured (np = 2) |
| --- | --- | --- | --- |
| \|E_dd − E_brute\| / N | 0 | 1e-9 eV/atom | **9.37e-16** |
| maxᵢ \|F_i,dd − F_i,brute\| | 0 | 1e-9 eV/Å | **9.71e-17** |
| ‖Σᵢ Fᵢ‖ | 0 | 1e-10 eV/Å | **6.30e-17** |
| `nswap` (min/max over steps) | 6 | exact | **6 / 6** |
| `Σ nrecv` vs `dd->n_ghost` | equal, > 0 | exact | **384 / 384**, 0 mismatches |
| full halo rebuilds (`halo_generation`) | ≥ 2 | — | **3** |
| \|E(np=2) − E(np=1)\| / \|E\| | 0 | 1e-12 | **5.33e-15** |
| owned rows assembled / lost-or-duplicated tags | 512 / 0 | exact | **512 / 0** |
| maxᵢ \|F_i,brute\| (non-vacuous) | > 1e-4 eV/Å | — | **8.94e-2** |

Reference energies of the step-21 configuration, for provenance (they are
**not** gated — the gate is the identity above, so the case survives any
future retune of ε/σ/box):

| Quantity | Value [eV] |
| --- | --- |
| E(np = 1), engine | −25.3535089627521160 |
| E(np = 2), engine | −25.3535089627519810 |
| E_brute, in-file O(N²) oracle (identical at np = 1 and np = 2) | −25.3535089627524606 |

The np = 1 and np = 2 **configurations** at step 21 are bitwise identical (the
oracle energy matches to the last bit); the ~1e-13 eV spread between the two
engine energies is pure summation order — decomposed half-pair sums plus
ghost rows versus one serial minimum-image walk.

At np = 1 the published `ctx.md.comm` is `nullptr` — the documented
null contract (no swap table exists on a single rank) — so the swap-table
gates are asserted on the np = 2 report only.

### Falsification (2026-08-25)

The gates were confirmed to bite, by perturbing the **in-file oracle** only
(no production code touched) and re-running:

| Injected defect | Result |
| --- | --- |
| oracle cutoff 0.99 × r_c | energy gate FAIL at 2.92e-4 eV/atom |
| oracle drops the x periodic image (the halo failure mode) | energy FAIL 3.22e-3 eV/atom **and** force FAIL 4.71e-3 eV/Å |

Both restored; the committed file is the unperturbed one.

### Scope note

This case pins **behaviour equivalence** of the swap-list halo at
`maxneed == 1`: single-hop swaps, dual-image ghosts across a 2-rank cut,
forward at both cadences, reverse exactly once. Multi-hop shells
(`maxneed > 1`) throw by contract and belong to `comm-topology-02`; rank
counts beyond 2 and any scaling curve belong to `comm-topology-03`.


## Reference — multi-hop halo == the single-image min-image sum (MPI DD)

oracle: literature
oracle_version: n/a (analytic identity; no oracle binary is run — the O(N²)
minimum-image sum is recomputed inside the regression. See citations below.)
perf_exempt: yes — this slug makes **no** performance claim. Multi-hop swaps
trade one wide message for `maxneed` forwarding rounds; quantifying that, and
any rank-scaling curve, belongs to `comm-topology-03`. No wall-clock number is
asserted here.

### Invariant under test

A domain decomposition is a **partition of one and the same sum**. Splitting a
periodic box across ranks may not change a single physical number — and that
must stay true when the ghost shell is *wider than a subdomain*, so no single
face exchange can fill it. The shell is then filled by `maxneed` forwarding
rounds: round `r` ships onward exactly what round `r − 1` received, one rank
further along the axis (Plimpton's `maxneed` / `sendlist` / `firstrecv`
construction). Every partial force accumulated on a ghost row must still return
to its owner **exactly once**, now unwinding the rounds deepest-hop-first.

Hence the criterion is an identity, not a remembered number:

| Quantity | Must equal |
| --- | --- |
| total potential energy after N steps | the flat O(N²) minimum-image sum over the same final configuration |
| every per-atom force | the same sum's analytic gradient |
| Σᵢ Fᵢ | 0 — non-zero means the reverse exchange ran zero times at some hop depth (force lost) or twice (energy pumped) |
| E(np = 4) | E(np = 1) — the decomposition is not a physical parameter |

The flat min-image sum is a *legal* oracle here because r_c = 6 Å < L/2 =
7.89 Å: each pair has exactly one image inside the cutoff no matter how the box
was cut. This is the same bound (`max_hops ≤ dims − 1`) that the swap-list
derivation still refuses to cross.

- S. Plimpton, *Fast Parallel Algorithms for Short-Range Molecular Dynamics*,
  J. Comput. Phys. **117**, 1–19 (1995), DOI 10.1006/jcph.1995.1039 — spatial
  decomposition, the ghost shell, and **hop-by-hop forwarding**: each round's
  message carries the data the previous round received, which is precisely the
  `maxneed > 1` path exercised here.
- A. P. Thompson et al., *LAMMPS — a flexible simulation tool for
  particle-based materials modeling at the atomic, meso, and continuum
  scales*, Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171 — `Comm::setup`'s `maxneed` / `nswap`
  derivation and the UNIFORM / NONUNIFORM layout semantics the ATV
  `Halo::derive_hops` reproduces, plus the "reverse exactly once" contract.

### System (child, np = 4 and np = 1)

64-atom simple-cubic LJ lattice, cubic box 15.78 Å, **forced 4×1×1
decomposition along x** (`ATV_FORCE_DIMS=4,1,1`; the override is ignored at
np = 1 because the product must equal the world size, which is what makes the
np = 1 leg a serial control rather than a second decomposed run).

| Parameter | Value |
| --- | --- |
| Sites | 4³ = 64, at `(i + 0.5) · a₀` — mid-slab, so no site sits on a partition plane |
| a₀ | 3.945 Å |
| Box | 15.78 Å cubic, periodic |
| ε | 0.0104 eV |
| σ | 3.4 Å |
| r_c | 6.0 Å |
| Verlet skin / reneighbor cadence | 0.6 Å / every 10 steps |
| Mass | 39.95 amu |
| dt / steps | 1.0 fs / 21 |
| Symmetry break | site (1, 1, 1) shifted 0.3 Å along (1,1,1)/√3 |

Pipeline — all defaults, the halo being the feature under test:
`PairLJCut{ε, σ, 6.0}` + `NeighborList{6.0, 0.6, every = 10}` +
`VelocityVerlet{1.0 fs}`. Halo / exchange / reverse live on `ctx.md.comm`
sized from `NeighborList::neighbor_reach()` (cutoff + skin = 6.6 Å).
Nothing here registers a halo component by hand, and nothing here sets a
hop count by hand — the round count is derived by `Halo::setup` from
the geometry.

#### Why this geometry

Subdomain width 15.78 / 4 = **3.945 Å** against a **6.6 Å** shell:

- `max_hops_x = ceil(6.6 / 3.945) = 2` — the shell **cannot** be filled by a
  single face exchange. Before `comm-topology-02` this configuration was
  refused outright ("shell wider than this rank's subdomain"); the whole point
  of the case is that it now runs *and is right*.
- `2 ≤ dims_x − 1 = 3` — still inside the minimum-image bound, so the flat
  O(N²) min-image sum is the correct oracle.
- At `dims_x = 4` the ranks `c − 2` and `c + 2` are the **same** rank, so the
  deepest hop delivers that rank's rows twice, as two distinct periodic images
  (ghost x-range `[sublo − 7.89, subhi + 7.89]`, i.e. 19.725 Å over a 15.78 Å
  box). Only one of the two images can be within r_c of a given owner because
  r_c < L/2, so the dual image is correct — and a "tidy-up" tag dedup would
  silently break it.
- y and z are undecomposed (`dims = 1`) ⇒ zero hops, one `sendself` placeholder
  pair each, and their periodicity is the pair kernel's minimum image. The
  swap table is therefore `2 · (2 + 1 + 1) = 8` swaps and every received ghost
  is the x shell.

21 steps at the every = 10 cadence cross **3** full rebuilds (`borders`), with
in-step forward refreshes in between, so both forward paths are covered at
both hop depths.

The oracle is computed **in this file**: an out-of-library `Component` at
`end_of_step @ +900` (after `mpi::Reduce` at −50 has published the globally
summed energy) allgathers every rank's owned rows by tag, reassembles the
global configuration, and evaluates the O(N²) minimum-image LJ sum and its
gradient. No third-party engine is imported, spawned, or read at build or run
time; no engine snapshot is stored.

### Pinned numbers (hard-coded gates in the regression)

Measured 2026-08-26 on the x86_64 login node (GCC + MPICH 4.3.2, Release,
`OMP_NUM_THREADS=1`); wall **0.72 s** for both child runs together
(budget ≤ 60 s).

| Gate | Want | Tolerance | Measured (np = 4) |
| --- | --- | --- | --- |
| \|E_dd − E_brute\| / \|E_brute\| | 0 | 1e-9 | **1.031e-15** |
| maxᵢ \|F_i,dd − F_i,brute\| | 0 | 1e-9 eV/Å | **2.320e-17** |
| observed x `max_hops` / deepest transmitting round | ≥ 2 / ≥ 2 | exact | **2 / 2** |
| \|E(np=4) − E(np=1)\| / \|E(np=1)\| | 0 | 1e-12 | **2.290e-16** |
| ‖Σᵢ Fᵢ‖ | 0 | 1e-10 eV/Å | **8.994e-18** |
| `dd->n_ghost` (min over steps) / `Σ nrecv` mismatches | > 0 / 0 | exact | **64 / 0** |
| full halo rebuilds (`halo_generation`) | ≥ 2 | — | **3** |
| owned rows assembled / lost-or-duplicated tags | 64 / 0 | exact | **64 / 0** |
| maxᵢ \|F_i,brute\| (non-vacuous) | > 1e-4 eV/Å | — | **1.453e-02** |

The hop row is what separates this slug from `comm-swaplist-halo-mpi`: it is
read back from the **published** `ctx.md.comm` twice over — the derived
`Halo::max_hops(0)` and the deepest `Swap::round` the table actually
materialised on a transmitting x swap — as the minimum over all 21 steps, so a
silent degradation to a single-hop shell on any one step fails the case
instead of hiding behind an energy that happens to still be right.

#### Pinned energy

Measured 2026-08-26, same environment. The **np = 1 energy is the pinned
literal**; the identity gates above are what the case enforces, so this pin is
provenance (it survives no retune of ε/σ/a₀ and is not itself gated).

| Quantity | Value [eV] |
| --- | --- |
| **E(np = 1), engine — the pin** | **−1.9390130589549823** |
| E(np = 4), engine | −1.9390130589549819 |
| E_brute, in-file O(N²) oracle (identical at np = 1 and np = 4) | −1.9390130589549839 |

#### Tolerance rationale

The two tolerances are not empirical windows; they follow from what the
decomposition *is*.

- **1e-9 relative energy / 1e-9 eV/Å force** — the engine and the oracle
  evaluate the **same** minimum-image sum, so the only admissible difference is
  floating-point summation order: the decomposed run adds half-pair
  contributions per rank plus ghost rows and MPI-reduces, the oracle walks one
  serial double loop. For 64 atoms in double precision that is O(N · ε_mach) ≈
  1e-15 relative, four orders inside the gate at 1e-9 — wide enough that a
  future reorder (xsimd, thread count, rank count) cannot flip the case, narrow
  enough that any *physical* defect fails it. Falsification below shows the
  smallest realistic defects land at 1e-2, seven orders past the gate.
- **1e-12 relative for E(np=4) vs E(np=1)** — tighter, because here both sides
  are the engine's own reduction of the identical trajectory; the only
  difference is the partition of the sum. Measured 2.29e-16, i.e. one ulp of
  the total.

Both are stricter than the generic 1e-6 numerical-energy tier of the harness,
deliberately: this quantity is an identity, not a converged approximation.

### Falsification (2026-08-26)

The gates were confirmed to bite, by perturbing the **in-file oracle** only
(no production code touched) and re-running:

| Injected defect | Result |
| --- | --- |
| oracle cutoff 0.99 × r_c | energy gate FAIL at 2.427e-2 relative |
| oracle drops the x periodic image (the multi-hop failure mode: the far ghosts never arrive) | energy FAIL 1.047e-1 relative **and** force FAIL 1.050e-2 eV/Å |

Both restored; the committed file is the unperturbed one.

### Scope note

This case pins the **correctness** of a `maxneed = 2` ghost shell on a uniform
brick: two forwarding rounds on x, dual-image ghosts from the wrap-around rank,
forward at both cadences, reverse exactly once per hop depth, and decomposition
invariance against the serial control. Out of scope, by spec: the non-uniform
(RCB) hop chain and the `max_hops > dims − 1` refusal (unit-tested serially in
`tests/cpu/communication/test_halo_maxneed.cpp`), deeper shells at 8
ranks (`tests/mpi/test_halo_dual_image_ghosts.cpp` `_n8`), and every
performance number (`comm-topology-03`).


## lammps-comm-brick-05-presets

oracle: literature
oracle_version: n/a (analytic identity; no oracle binary is run — the O(N²)
minimum-image sum is recomputed inside the regression)
perf_exempt: yes — this slug makes **no** performance claim.

### Invariant under test

`cpu::Evaluate` + `PairLJCut` + `NeighborList` under MPI domain
decomposition reproduces the flat O(N²) minimum-image LJ sum. Halo /
exchange / reverse live on `ctx.md.comm`; the compiled Plan contains
`reduce` and does not contain `halo_send` / `halo_recv` / `migrate` /
`halo_refresh_*`.

\[
\bigl|E_{\mathrm{dd}} - E_{\mathrm{brute}}\bigr| / |E_{\mathrm{brute}}|
\le 10^{-9}.
\]

- S. Plimpton, J. Comput. Phys. **117**, 1–19 (1995),
  DOI 10.1006/jcph.1995.1039
- A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171
- A. Rahman, Phys. Rev. **136**, A405 (1964),
  DOI 10.1103/PhysRev.136.A405 — LJ argon parameters


## Reference — wide-box parity: a multi-hop, charged, bonded decomposition == the serial direct sum

oracle: literature
oracle_version: n/a (analytic identity; no oracle binary is run — the O(N²)
minimum-image direct sum is recomputed inside the regression. See citations
below.)
perf_exempt: yes — this slug makes **no** performance claim. Rank-scaling,
steps/s and the LAMMPS comparison of `comm-topology-03` are measured with
`benchmarks/compare/gaff-peo-lammps` and are explicitly not acceptance
criteria. No wall-clock number is asserted here.

### Invariant under test

A domain decomposition is a **partition of one and the same sum**. Splitting a
periodic box across ranks may not change a single physical number — and that
must stay true when

1. the ghost shell is **wider than a subdomain** on one decomposed axis, so no
   single face exchange can fill it (the shell is filled by `maxneed`
   forwarding rounds: round `r` ships onward exactly what round `r − 1`
   received, one rank further along the axis — Plimpton's `maxneed` /
   `sendlist` / `firstrecv` construction), **and**
2. a second axis is decomposed at `dims = 2`, where the `+d` and `−d` partner
   are the **same** rank and every atom of the face overlap band ships through
   both faces as two distinct periodic images, **and**
3. the rows in flight carry more than positions: **charges** (the cut-Coulomb
   kernel reads `kAtom_charge` on ghost rows) and the **bonded extension**
   (bond / angle tuples whose partner is a ghost across a cut, resolved through
   the same `tag → local` map, together with the special-bond exclusion lists).

Hence the criterion is an identity, not a remembered number, and it is gated in
two stages:

| Stage | Quantity | Must equal |
| --- | --- | --- |
| 1 (np = 1) | total and per-term potential energy after 21 steps | the flat O(N²) minimum-image direct sum over the same final configuration, with the special-bond exclusions mirrored |
| 1 (np = 1) | every per-atom force | the same sum's analytic gradient |
| 2 (np = 8) | E(np = 8, 4×2×1) | E(np = 1) — the decomposition is not a physical parameter |
| 2 (np = 8) | every per-atom force, tag-aligned | F(np = 1) |
| 2 (np = 8) | Σᵢ Fᵢ | 0 — non-zero means the reverse exchange ran zero times at some hop depth (force lost) or twice (energy pumped) |

Stage 1 is what makes stage 2 meaningful: it proves the in-file oracle really
is this model Hamiltonian (including the exclusion rule), so stage 2's parity
is a statement about the decomposition and not about two engines agreeing on
the same mistake.

The flat min-image sum is a *legal* oracle here because r_c = 6 Å ≤ L_d/2 on
every axis (12 / 7 / 7 Å): each pair has exactly one image inside the cutoff no
matter how the box was cut. That is the same bound (`max_hops ≤ dims − 1`) the
swap-list derivation still refuses to cross.

- M. P. Allen & D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford University Press (2017), §1.6.3 — the minimum-image convention and the
  direct pair sum that **defines** a cutoff model Hamiltonian; this is the
  oracle evaluated in the file.
- S. Plimpton, *Fast Parallel Algorithms for Short-Range Molecular Dynamics*,
  J. Comput. Phys. **117**, 1–19 (1995), DOI 10.1006/jcph.1995.1039 — spatial
  decomposition, the `r_c + skin` ghost-shell condition, and **hop-by-hop
  forwarding**: each round's message carries what the previous round received,
  precisely the `maxneed > 1` path exercised on x here.
- A. P. Thompson et al., *LAMMPS — a flexible simulation tool …*, Comput. Phys.
  Commun. **271**, 108171 (2022), DOI 10.1016/j.cpc.2021.108171 — `Comm::setup`'s
  `maxneed` / `nswap` derivation, the "reverse exactly once" contract, and the
  `special_bonds` 1-2 / 1-3 / 1-4 weighting convention mirrored by the oracle.

### System (child, np = 1 and np = 8)

32 bent triatomic molecules (96 atoms) on a body-centred molecular lattice in a
**24 × 14 × 14 Å** orthorhombic periodic box, under a forced **4×2×1**
decomposition (`ATV_FORCE_DIMS=4,2,1`; the override is ignored at np = 1
because the product must equal the world size, which is what makes the np = 1
leg a serial control rather than a second decomposed run).

| Parameter | Value |
| --- | --- |
| Molecules / atoms | 32 / 96 (A–B–A, atom order = molecule-major, tags 1-based) |
| Lattice | 4 × 2 × 2 cells of 6.0 × 7.0 × 7.0 Å, corner + body-centre basis |
| Lattice phase | (0.13, 0.07, 0.11) Å — puts the corner sublattice just off the partition planes, so those molecules **straddle** a cut while no atom lands on one |
| Orientation | per-molecule Rz(0.7 m + 0.3) · Ry(0.37 m + 1.1); centroid (not the centre atom) on the lattice site |
| Bond / angle offsets | r = 1.5 + 0.08 sin(1.7 m) Å, θ = 112° + 0.12 cos(1.3 m) rad — every bonded term off equilibrium, deterministic, no RNG |
| Charges | q_B = +0.70 e, q_A = −0.35 e ⇒ net 0 per molecule and globally |
| Masses | B 15.999, A 12.011 amu |
| LJ (one type) | ε = 0.0104 eV, σ = 3.0 Å, r_c = 6.0 Å, energy-shifted |
| Coulomb | `PairCoulCut`, r_c = 6.0 Å, unshifted, 14.399645 eV·Å/e² |
| special_bonds | 0.0 / 0.0 / 0.5 (Amber) on **both** pair styles |
| Bond / angle | K = 12.0 eV/Å², r₀ = 1.5 Å; K = 3.0 eV/rad², θ₀ = 1.9548 rad |
| Verlet skin / reneighbor cadence | 0.6 Å / every 10 steps |
| dt / steps | 0.5 fs / 21 (velocities start at zero) |

Pipeline — all defaults, the halo being the feature under test:
`PairLJCut` + `PairCoulCut` + `BondHarmonic` + `AngleHarmonic` +
`NeighborList{6.0, 0.6, every = 10}` + `VelocityVerlet{0.5 fs}`.
`MD::install_defaults()` injects `comm::CommBrick` with a
**max(1.1 · 6.0, 6.0 + 0.6) = 6.6 Å** shell at `pre_neighbor @ 0` and
`comm::CommBrick` at `post_force @ −1000`. Nothing here registers a halo
component by hand and nothing sets a hop count by hand — the round count is
derived by `Halo::setup` from the geometry. **No PPPM**: the build needs
only `ATV_BUILD_MD` + `ATV_USE_MPI`, no FFTW gate.

#### Why this geometry

| Axis | dims | span | Shell 6.6 Å ⇒ | Branch exercised |
| --- | --- | --- | --- | --- |
| x | 4 | 6.0 Å | `max_hops = ⌈6.6/6.0⌉ = 2` (≤ dims − 1 = 3) | **multi-hop** forwarding |
| y | 2 | 7.0 Å | `max_hops = 1`, but 2 · 6.6 = 13.2 > 7.0 | **dual-image** ghosts from one rank through both faces |
| z | 1 | 14 Å | 0 hops | pair-kernel minimum image |

A **12 Å** y edge (span 6.0 Å) — the naive "24 × 12 × 12" reading of the
spec — is *not* a legal 2-hop y axis and is not what this case runs: at
`dims_y = 2` the deepest reachable hop is `dims − 1 = 1`, because round 2 would
wrap onto this rank's own periodic image. `Halo::derive_hops`
(`src/cpu/communication/halo.cpp`) refuses it outright:

```
Halo: the requested ghost shell (6.600000 A) violates the minimum-image
convention on periodic axis 1: ... The bound is shell <= L - w_self =
12.000000 - 6.000000 = 6.000000 A.
```

So a 2-hop x axis and a 2-hop y axis at `dims_y = 2` cannot coexist by
construction; **7.0 Å is the narrowest legal y span that still overlaps its own
two faces**, which is why the box is 24 × 14 × 14 Å. The same argument fixes
the geometry of the unit-level companion
(`tests/mpi/test_halo_dual_image_ghosts.cpp`, case
`ProductionGridTwoNarrowAxesMatchBruteForce`).

What this slug adds over `regressions/md/domain-decomposition` (LJ only,
4×1×1): the multi-hop shell here must also carry **ghost charges** and the
**bonded extension** across the cuts (half the molecules straddle a partition
plane by construction), and the special-bond exclusion lists must resolve
through those same ghost rows — the shape of the 24k GAFF target system.

21 steps at the every = 10 cadence cross **3** full rebuilds (`borders`), with
in-step forward refreshes in between, so both forward paths are covered at both
hop depths.

The oracle is computed **in this file**: an out-of-library `Component` at
`end_of_step @ +900` (after `mpi::Reduce` at −50 has published the globally
summed energies) allgathers every rank's owned rows by tag, reassembles the
global configuration, and evaluates the O(N²) minimum-image direct sum and its
gradient — LJ shifted exactly as `PairLJCut` shifts it, Coulomb unshifted
exactly as `PairCoulCut` computes it, harmonic bond and the LAMMPS
harmonic-angle gradient, with same-molecule pairs dropped because with weights
(0, 0, 0.5) every intramolecular pair of a 3-site A–B–A molecule is 1-2 or 1-3.
(That topology contains **no** 1-4 pair, so the 0.5 weight is inert here; the
exclusion the gate actually tests is the 1-2 / 1-3 drop, and falsification C
below shows it is load-bearing.) No third-party engine is imported, spawned or
read at build or run time; no engine snapshot is stored.

### Pinned numbers (hard-coded gates in the regression)

Measured 2026-08-26 on the x86_64 login node (GCC + MPICH 4.3.2, Release,
`OMP_NUM_THREADS=1`); wall **0.97 s** for both child runs together
(budget ≤ 60 s).

| Gate | Want | Tolerance | Measured |
| --- | --- | --- | --- |
| \|E(np1) − E_oracle\| / \|E_oracle\| | 0 | 1e-9 | **4.252e-16** |
| … pair (LJ + Coulomb) share | 0 | 1e-9 | **4.428e-16** |
| … bond share | 0 | 1e-9 | **7.812e-16** |
| … angle share | 0 | 1e-9 | **1.479e-16** |
| maxᵢ \|F_i(np1) − F_i,oracle\| | 0 | 1e-9 eV/Å | **7.216e-15** |
| \|E(np8) − E(np1)\| / \|E(np1)\| | 0 | 1e-10 | **1.417e-16** |
| max \|F_i(np8) − F_i(np1)\| (tag-aligned) | 0 | 1e-9 eV/Å | **5.894e-14** |
| \|E(np8) − E_oracle\| / \|E_oracle\| | 0 | 1e-9 | **5.669e-16** |
| maxᵢ \|F_i(np8) − F_i,oracle\| | 0 | 1e-9 eV/Å | **3.989e-14** |
| ‖Σᵢ Fᵢ‖ (np = 8) | 0 | 1e-10 eV/Å | **3.168e-16** |
| x `max_hops` / deepest transmitting round / y `max_hops` | ≥ 2 / ≥ 2 / = 1 | exact | **2 / 2 / 1** |
| `dd->n_ghost` (min over steps) / `Σ nrecv` mismatches | > 0 / 0 | exact | **163 / 0** |
| full halo rebuilds (`halo_generation`) | ≥ 2 | — | **3** |
| owned rows assembled / lost-or-duplicated tags | 96 / 0 | exact | **96 / 0** |
| maxᵢ \|F_i,oracle\| (non-vacuous) | > 1e-4 eV/Å | — | **1.146e+00** |
| \|E_pair\| / \|E_bond\| / \|E_angle\| (non-vacuous) | > 1e-4 eV | — | **1.203e+01 / 1.244e-01 / 3.752e-01** |

The hop row is read back from the **published** `ctx.md.comm` twice over —
the derived `Halo::max_hops(0)` and the deepest `Swap::round` the table
actually materialised on a transmitting x swap — as the minimum over all 21
steps, so a silent degradation to a single-hop shell on any one step fails the
case instead of hiding behind an energy that happens to still be right. The
y row is pinned to exactly 1 for the same reason in the other direction: a
"2" there would mean the minimum-image bound stopped being enforced.

#### Pinned energy

Measured 2026-08-26, same environment. The **np = 1 energy is the pinned
literal**; the identity gates above are what the case enforces, so this pin is
provenance (it does not survive a retune of ε/σ/q/K and is not itself gated).

| Quantity | Value [eV] |
| --- | --- |
| **E(np = 1), engine — the pin** | **12.5344130123646504** |
| … pair (LJ + cut Coulomb) | 12.0348248656285470 |
| … bond | 0.1243554236591224 |
| … angle | 0.3752327230769815 |
| E(np = 8), engine | 12.5344130123646522 |
| E_oracle, in-file O(N²) direct sum | 12.5344130123646451 |

#### Tolerance rationale

The tolerances are not empirical windows; they follow from what the
decomposition *is*.

- **1e-9 relative energy / 1e-9 eV/Å force vs the oracle** — the engine and the
  oracle evaluate the **same** minimum-image sum of the **same** Hamiltonian,
  so the only admissible difference is floating-point **summation order**: the
  engine walks a cell/Verlet list with per-thread and per-rank partial sums and
  MPI-reduces, the oracle walks one serial double loop. For 96 atoms in double
  precision that is O(N · ε_mach) ≈ 1e-15 relative — measured 4e-16 — six
  orders inside the gate, wide enough that a future reorder (xsimd, thread
  count, rank count) cannot flip the case, narrow enough that any *physical*
  defect fails it. The falsification table below shows the smallest realistic
  defects land at 1e-1, eight orders past the gate.
- **1e-10 relative for E(np=8) vs E(np=1)** — tighter, because here both sides
  are the engine's own reduction of the identical trajectory; the only
  difference is the partition of the sum. Measured 1.4e-16, i.e. one ulp of the
  total.
- **1e-9 eV/Å for the tag-aligned force parity** — a per-component absolute
  window on forces of order 1 eV/Å, i.e. ~1e-9 relative. Measured 5.9e-14: the
  extra decades over the energy figure are the accumulated reassociation of
  ghost-force reverse exchange at two hop depths, still five orders inside the
  gate.

All are stricter than the generic 1e-6 numerical-energy tier of the harness,
deliberately: these quantities are identities, not converged approximations.

### Falsification (2026-08-26)

The gates were confirmed to bite, by perturbing the **in-file oracle** only (no
production code touched, no gate loosened) and re-running:

| Injected defect | Result |
| --- | --- |
| A. oracle cutoff 0.99 × r_c | energy FAIL at 3.915e-1 relative, force FAIL at 3.034e-1 eV/Å (5 gates) |
| B. oracle drops the x periodic image (the multi-hop failure mode: the far ghosts never arrive) | energy FAIL 9.979e-1 relative, force FAIL 5.359e+2 eV/Å (7 gates) |
| C. oracle forgets the special-bond exclusion (1-2 / 1-3 pairs at ~1.5 Å re-enter both pair styles) | pair-term FAIL 9.989e-1 relative, force FAIL 1.832e+3 eV/Å (5 gates) |

In every case the **np = 8 vs np = 1** rows stayed green, as they must — those
two legs share the oracle-independent identity — which is exactly why stage 1
exists: only stage 1 can catch an oracle that has stopped describing the model.
All three perturbations were reverted; the committed file is the unperturbed
one and re-runs to the pinned numbers above.

### Scope note

This case pins the **correctness** of a `maxneed = 2` ghost shell carrying
charges and bonded topology on a two-axis brick: two forwarding rounds on x,
dual-image ghosts on y, forward at both cadences, reverse exactly once per hop
depth, and decomposition invariance against the serial control. Out of scope,
by spec: the non-uniform (RCB) hop chain and the `max_hops > dims − 1` refusal
(unit-tested serially in `tests/cpu/communication/test_halo_maxneed.cpp`),
deeper shells at 8 ranks on one axis
(`tests/mpi/test_halo_dual_image_ghosts.cpp` `_n8`), long-range electrostatics
(no PPPM here), the 24k-atom / 256-rank real-machine run
(`benchmarks/compare/gaff-peo-lammps/widebox_parity.sh`), and every performance number.
