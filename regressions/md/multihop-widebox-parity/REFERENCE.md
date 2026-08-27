# Reference — wide-box parity: a multi-hop, charged, bonded decomposition == the serial direct sum

oracle: literature
oracle_version: n/a (analytic identity; no oracle binary is run — the O(N²)
minimum-image direct sum is recomputed inside the regression. See citations
below.)
perf_exempt: yes — this slug makes **no** performance claim. Rank-scaling,
steps/s and the LAMMPS comparison of `comm-topology-03` live in the dated
snapshot block of `tools/bench_gaff_lammps/README.md` and are explicitly not
acceptance criteria. No wall-clock number is asserted here.

## Invariant under test

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

## System (child, np = 1 and np = 8)

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

### Why this geometry

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

What this slug adds over `regressions/md/comm-topology-multihop` (LJ only,
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

## Pinned numbers (hard-coded gates in the regression)

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

### Pinned energy

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

### Tolerance rationale

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

## Falsification (2026-08-26)

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

## Scope note

This case pins the **correctness** of a `maxneed = 2` ghost shell carrying
charges and bonded topology on a two-axis brick: two forwarding rounds on x,
dual-image ghosts on y, forward at both cadences, reverse exactly once per hop
depth, and decomposition invariance against the serial control. Out of scope,
by spec: the non-uniform (RCB) hop chain and the `max_hops > dims − 1` refusal
(unit-tested serially in `tests/cpu/communication/test_halo_maxneed.cpp`),
deeper shells at 8 ranks on one axis
(`tests/mpi/test_halo_dual_image_ghosts.cpp` `_n8`), long-range electrostatics
(no PPPM here), the 24k-atom / 256-rank real-machine run
(`tools/bench_gaff_lammps/widebox_parity.sh`), and every performance number.
