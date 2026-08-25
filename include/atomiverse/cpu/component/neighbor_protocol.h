/// @file cpu/component/neighbor_protocol.h
/// @brief NeighborConsumer / NeighborProvider — opt-in protocols for the
///        neighbor-list contract. CUDA-free header.
///
/// Split out of `Component` (arch-refactor-03-neighbor), following the
/// `HaloParticipant` / `PairEvaluator` opt-in precedent: the consumer side
/// (pair styles, real-space electrostatics) and the supply side
/// (`cpu::md::NeighborList`) are disjoint populations, so they get two
/// independent protocols — mirroring the halo protocol's send/recv split.
/// A component that participates inherits the protocol alongside
/// `Component` and overrides the matching `as_*` accessor to return
/// `this`; the driver infrastructure scan probes those typed accessors —
/// no `dynamic_cast`, no `name()` matching.
#pragma once

#include "atomiverse/core/precision.h"

namespace ATV {
namespace cpu {

/// @brief Consumer side of the neighbor-list contract: a component that
///        requires a neighbor structure covering its interaction cutoff.
///
/// The category drivers throw at `build()` when a consumer is registered
/// without a neighbor structure, and size the auto-injected MPI halo from the
/// widest consumer halo requirement (padded 1.1×) or, when a provider declares
/// one, from `NeighborProvider::neighbor_reach()` — whichever is larger.
class NeighborConsumer {
  public:
    virtual ~NeighborConsumer() = default;

    /// @brief Interaction cutoff the neighbor list must cover [Å].
    ///
    /// Return `0` to declare "consumer without a cutoff" — the missing-NL
    /// gate still applies, but halo sizing skips this component (the pre-protocol
    /// zero-hint semantics).
    /// @return Cutoff radius [Å]; `0` = no cutoff declared.
    virtual AccReal neighbor_cutoff() const = 0;

    /// @brief How far the MPI halo must reach for this component [Å].
    ///
    /// Defaults to the interaction cutoff, which is right for any style whose atoms
    /// only ever need their OWN neighbours: a pair style touches `j` within `r_c` of
    /// an owned `i`, and a ghost shell of `r_c` supplies every such `j`.
    ///
    /// A **many-body** style needs more. EAM's second pass wants `F'(rho_j)` for a
    /// ghost `j`, and `rho_j` is a sum over *j's* neighbours — so completing it
    /// locally requires the neighbours of the ghosts, i.e. a **ghost-of-ghost** shell
    /// of `2 r_c`. Declaring that here buys the whole thing without a single extra
    /// message: the engine exchanges exactly one forward halo (`pre_neighbor`) and one
    /// reverse halo (`post_force`), and there is no communication point inside the
    /// force phase in which a derived per-atom quantity could be traded. A wider halo
    /// sidesteps that hole instead of punching a new one through it.
    ///
    /// @return Halo radius [Å]; `0` = no halo requirement declared.
    virtual AccReal halo_cutoff() const { return neighbor_cutoff(); }
};

/// @brief Supply side of the neighbor-list contract: a component that
///        builds and publishes the neighbor structure.
///
/// The publication channel stays `Context::pair_list` (a non-owning pointer
/// written in the provider's `build()`/`launch()`), not a virtual on this
/// interface. The one thing the interface does carry is the row-order
/// guarantee below, because a provider is the only component that renumbers
/// rows and a driver has to be able to forbid it.
class NeighborProvider {
  public:
    virtual ~NeighborProvider() = default;

    /// @brief Forbid the spatial reorder of owned rows for the rest of this
    ///        provider's life (LAMMPS `atom_modify sort`).
    ///
    /// **Monte Carlo requires this.** An MC trial move is a transaction whose
    /// changeset is a set of frame ROWS (`mc::Transaction` records "row `r`
    /// looked like this"), and the neighbour rebuild runs *inside* the trial —
    /// the list is bound at `sampling::Phase::energy`, ahead of the evaluator
    /// that scores the proposal. A reorder landing between `record_touch` and
    /// `rollback` therefore splices a saved row over a completely unrelated
    /// atom. Two atoms end up on top of each other, the sampler's bookkeeping
    /// energy stops describing the configuration it actually holds, and every
    /// later trial is rejected against an overlap it cannot see: the Markov
    /// chain freezes at whatever state it was in, with no error and no NaN.
    ///
    /// A spatial sort is a cache optimisation; correctness of the sampler is
    /// not negotiable against it.
    virtual void forbid_row_reordering() = 0;

    /// @brief Total distance out to which this provider enumerates pairs [Å].
    ///
    /// A Verlet structure does not stop at the interaction cutoff: it builds out
    /// to `cutoff + skin`, and that band is the entire reason the list is reusable
    /// between rebuilds. Any pair inside it is *supposed* to be enumerated, so the
    /// MPI ghost shell has to be at least this wide. A shell sized from the
    /// interaction cutoff alone leaves a cross-rank `j` sitting in the band with no
    /// ghost on `i`'s rank: the pair is absent from the CSR outright rather than
    /// merely inactive, and stays absent until the next full rebuild while the
    /// displacement trigger (`max_disp * 2 > skin`) is by construction not watching
    /// for it. The pair can then drift from the band down below the interaction
    /// cutoff contributing exactly zero force the whole way.
    ///
    /// The driver infrastructure scan reads this and sizes the auto-injected halo
    /// from it (max over providers).
    ///
    /// @return Enumeration radius [Å]; `0` = no reach declared (the halo falls
    ///         back to the consumer-cutoff estimate).
    virtual AccReal neighbor_reach() const { return AccReal(0); }
};

} // namespace cpu
} // namespace ATV
