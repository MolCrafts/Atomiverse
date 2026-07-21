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
/// without a neighbor structure, and size the auto-injected MPI halo from
/// the first consumer's cutoff (skin-padded, 1.1×).
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
/// Pure tag protocol — the publication channel stays `Context::pair_list`
/// (a non-owning pointer written in the provider's `build()`/`launch()`),
/// not a virtual on this interface.
class NeighborProvider {
  public:
    virtual ~NeighborProvider() = default;
};

} // namespace cpu
} // namespace ATV
