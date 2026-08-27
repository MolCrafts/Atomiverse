# Reference — multi-hop halo == the single-image min-image sum (MPI DD)

oracle: literature
oracle_version: n/a (analytic identity; no oracle binary is run — the O(N²)
minimum-image sum is recomputed inside the regression. See citations below.)
perf_exempt: yes — this slug makes **no** performance claim. Multi-hop swaps
trade one wide message for `maxneed` forwarding rounds; quantifying that, and
any rank-scaling curve, belongs to `comm-topology-03`. No wall-clock number is
asserted here.

## Invariant under test

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

## System (child, np = 4 and np = 1)

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

### Why this geometry

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

## Pinned numbers (hard-coded gates in the regression)

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

### Pinned energy

Measured 2026-08-26, same environment. The **np = 1 energy is the pinned
literal**; the identity gates above are what the case enforces, so this pin is
provenance (it survives no retune of ε/σ/a₀ and is not itself gated).

| Quantity | Value [eV] |
| --- | --- |
| **E(np = 1), engine — the pin** | **−1.9390130589549823** |
| E(np = 4), engine | −1.9390130589549819 |
| E_brute, in-file O(N²) oracle (identical at np = 1 and np = 4) | −1.9390130589549839 |

### Tolerance rationale

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

## Falsification (2026-08-26)

The gates were confirmed to bite, by perturbing the **in-file oracle** only
(no production code touched) and re-running:

| Injected defect | Result |
| --- | --- |
| oracle cutoff 0.99 × r_c | energy gate FAIL at 2.427e-2 relative |
| oracle drops the x periodic image (the multi-hop failure mode: the far ghosts never arrive) | energy FAIL 1.047e-1 relative **and** force FAIL 1.050e-2 eV/Å |

Both restored; the committed file is the unperturbed one.

## Scope note

This case pins the **correctness** of a `maxneed = 2` ghost shell on a uniform
brick: two forwarding rounds on x, dual-image ghosts from the wrap-around rank,
forward at both cadences, reverse exactly once per hop depth, and decomposition
invariance against the serial control. Out of scope, by spec: the non-uniform
(RCB) hop chain and the `max_hops > dims − 1` refusal (unit-tested serially in
`tests/cpu/communication/test_halo_maxneed.cpp`), deeper shells at 8
ranks (`tests/mpi/test_halo_dual_image_ghosts.cpp` `_n8`), and every
performance number (`comm-topology-03`).
