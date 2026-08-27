# Reference — ghost-charge correctness after the const-pot Q solve (MPI DD)

oracle: literature  
oracle_version: n/a (self-consistency invariant; no oracle binary is run —
see citations below; perf_exempt)

## Invariant under test

Constant-potential MD rewrites electrode charges **mid-step** (the Q solve at
`pre_force`), i.e. after the step's forward halo has populated ghost atoms.
Under MPI domain decomposition the updated owner charges must be
**forward-communicated to every ghost copy before the force stages read
them** — otherwise pair/k-space Coulomb forces at rank boundaries use charges
that are one solve stale (and identically zero on the first step, since
electrode charges start at 0).

This is the documented contract of the LAMMPS ELECTRODE implementation: after
every conp charge solve, `fix electrode` issues a forward communication of the
per-atom charge so ghosts match owners before force evaluation.

- A. P. Thompson et al., *LAMMPS — a flexible simulation tool for
  particle-based materials modeling at the atomic, meso, and continuum
  scales*, Comput. Phys. Commun. **271**, 108171 (2022),
  DOI 10.1016/j.cpc.2021.108171 — forward/reverse ghost communication
  semantics (owner → ghost refresh of per-atom state).
- L. J. V. Ahrens-Iwers, M. Janssen, S. R. Tee, R. H. Meißner,
  *ELECTRODE: An electrochemistry package for atomistic simulations*,
  J. Chem. Phys. **157**, 084801 (2022), DOI 10.1063/5.0099239 —
  constant-potential charge solve per step; solved charges are
  forward-communicated (`comm->forward_comm`) before force evaluation.
- A. Marin-Laflèche et al., *MetalWalls: A classical molecular dynamics
  software dedicated to the simulation of electrochemical systems*,
  J. Open Source Softw. **5**, 2373 (2020), DOI 10.21105/joss.02373 —
  the MetalWalls constant-potential loop this capability maps.

## System (child, np = 2)

38-atom parallel-plate capacitor, box 20 × 10 × 16 Å (orthorhombic, forced
2×1×1 decomposition along x so each rank owns part of BOTH plates):

| Species | Type | Count | z [Å] | q₀ [e] |
| --- | --- | --- | --- | --- |
| Bottom plate (Ψ = 0 V) | 1 | 15 (5×3 grid) | 5 | 0 (solved) |
| Top plate (Ψ = 1 V) | 2 | 15 (5×3 grid) | 11 | 0 (solved) |
| Electrolyte ions | 3 | 8 | 7–9.5 | ±0.25 (neutral) |

Pipeline (all defaults — the packer refresh is the feature under
test): `VelocityVerlet{0.1 fs}` + `NeighborList{3.0, 0.3}` +
`PairCoulCut{3.0}` + `ConstPotBO{η = 2 Å⁻¹}` (electroneutrality on, matrix
rebuilt per step). Halo / exchange / reverse live on `ctx.md.comm`;
`ConstPotBO` posts `ChargePacker` after writing owned charges. 5 MD steps.

An out-of-library `GhostChargeChecker` Component (defined in the regression
file — the user-extension path) runs at the force phase and compares every
ghost row's charge against its owner's current value (tag-matched via
allgather).

## Pinned numbers (hard-coded in the regression)

| Gate | Value | Why |
| --- | --- | --- |
| max \|q_ghost − q_owner\| over all ghosts, ranks, steps | **0** (tol 1e-15 e) | halo transport is a memcpy of the owner value — bitwise |
| electrode ghosts with \|q\| > 1e-12 e | **> 0** | non-vacuous: electrode charges start at 0; only the refreshed solve makes ghosts nonzero |
| unresolved ghost tags | **0** | every ghost must map to an owner |
| checker launches | ≥ 5 (one per step) | invariant enforced on every step |

Without the auto-injected charge refresh, gate (a) fails with
max |dq| ≈ the solved electrode charge magnitude (~1e-2 e in this geometry)
at the first step — the stale-ghost failure mode this regression pins.

## Scope note

The per-rank `ConstPotBO` solve under DD is a rank-local partial solve (the
DD-aware global electrode solve is a separate queued capability). This
regression pins the **transport** invariant only: whatever charge the owner
solved, every ghost copy carries it before forces read it. Runtime ≈ seconds
(≤ 60 s).
