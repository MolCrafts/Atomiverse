# Reference — swap-list halo == the single-image min-image sum (MPI DD)

oracle: literature
oracle_version: n/a (analytic identity; no oracle binary is run — the O(N²)
minimum-image sum is recomputed inside the regression. See citations below.)
perf_exempt: yes — this slug makes **no** performance claim. The swap-list
refactor replaces per-step, per-face geometry re-derivation with a table
derived once per rebuild; quantifying that is a separate slug
(`comm-topology-03`), and no wall-clock number is asserted here.

## Invariant under test

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

## System (child, np = 2 and np = 1)

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
`VelocityVerlet{1.0 fs}`. `MD::install_defaults()` injects `comm::HaloSend`
with a **9.35 Å** shell (1.1 × the pair cutoff) at `pre_neighbor @ 0` and
`comm::HaloRecv` at `post_force @ -1000`. Nothing here registers a halo
component by hand.

### Why this geometry

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

## Pinned numbers (hard-coded gates in the regression)

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

At np = 1 the published `ctx.md.swap_list` is `nullptr` — the documented
null contract (no swap table exists on a single rank) — so the swap-table
gates are asserted on the np = 2 report only.

## Falsification (2026-08-25)

The gates were confirmed to bite, by perturbing the **in-file oracle** only
(no production code touched) and re-running:

| Injected defect | Result |
| --- | --- |
| oracle cutoff 0.99 × r_c | energy gate FAIL at 2.92e-4 eV/atom |
| oracle drops the x periodic image (the halo failure mode) | energy FAIL 3.22e-3 eV/atom **and** force FAIL 4.71e-3 eV/Å |

Both restored; the committed file is the unperturbed one.

## Scope note

This case pins **behaviour equivalence** of the swap-list halo at
`maxneed == 1`: single-hop swaps, dual-image ghosts across a 2-rank cut,
forward at both cadences, reverse exactly once. Multi-hop shells
(`maxneed > 1`) throw by contract and belong to `comm-topology-02`; rank
counts beyond 2 and any scaling curve belong to `comm-topology-03`.
