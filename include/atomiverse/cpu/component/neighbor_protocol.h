/// @file cpu/component/neighbor_protocol.h
/// @brief NeighborConsumer / NeighborProvider — opt-in protocols for the
///        neighbor-list contract. CUDA-free header.
///
/// Split out of `Component` (arch-refactor-03-neighbor), following the
/// `PairEvaluator` opt-in precedent: the consumer side
/// (pair styles, real-space electrostatics) and the supply side
/// (`cpu::md::NeighborList`) are disjoint populations, so they get two
/// independent protocols — mirroring the halo protocol's send/recv split.
/// A component that participates inherits the protocol alongside
/// `Component` and overrides the matching `as_*` accessor to return
/// `this`; the driver infrastructure scan probes those typed accessors —
/// no `dynamic_cast`, no `name()` matching.
// Per-consumer half/full requests follow LAMMPS src/neigh_request.h,
// a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b. Copyright (2003) Sandia
// Corporation. Under Contract DE-AC04-94AL85000, the U.S. Government retains
// certain rights. Distributed under the GNU General Public License.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/precision.h"

namespace ATV {
namespace cpu {

class Frame;
struct Context;

/// Each consumer selects its counting convention independently.
enum class NeighborListMode { Half, Full };

/// @brief Consumer side of the neighbor-list contract: a component that
///        requires a neighbor structure covering its interaction cutoff.
///
/// The category drivers throw at `build()` when a consumer is registered
/// without a neighbor structure. `Runtime` sizes `ctx.md.comm` from
/// `NeighborProvider::neighbor_reach()`.
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

    virtual NeighborListMode neighbor_list_mode() const {
        return NeighborListMode::Half;
    }

    /// @brief Require neighbour rows centred on ghosts as well as owners.
    /// Independent of full/half pair counting; ordinary full lists (e.g. SW)
    /// only need owned centres. Currently supported for full lists.
    virtual bool requires_ghost_neighbor_rows() const { return false; }

    /// @brief How far the MPI halo must reach for this component [Å].
    ///
    /// Defaults to the interaction cutoff, which is right for any style whose atoms
    /// only ever need their OWN neighbours: a pair style touches `j` within `r_c` of
    /// an owned `i`, and a ghost shell of `r_c` supplies every such `j`.
    ///
    /// A consumer may require a wider coordinate dependency independently of
    /// pair counting or ghost-centred rows. EAM exchanges density/derivatives
    /// and SW enumerates triplets at owned centres; both need only r_c.
    ///
    /// @return Halo radius [Å]; `0` = no halo requirement declared.
    virtual AccReal halo_cutoff() const { return neighbor_cutoff(); }
};

/// @brief Supply side of the neighbor-list contract: a component that
///        builds and publishes the neighbor structure.
///
/// Publishes independent half/full views in Context. Providers receive the
/// current consumer requirements and the driver's row-order constraints.
class NeighborProvider {
  public:
    virtual ~NeighborProvider() = default;

    /// Publish this provider's borrowed views/cadence for the active Plan.
    virtual void activate(Context&) {}

    /// Compile the union of current consumers; replacement, not sticky opt-in.
    /// A provider supporting only half lists must reject a full-list request.
    virtual void configure_lists(bool half, bool full,
                                 bool ghost_rows = false) {
        (void)half;
        if (full || ghost_rows)
            throw std::runtime_error(
                "neighbor provider does not support full lists");
    }

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

    /// @brief Largest interaction cutoff safely supported during list reuse.
    /// @return Interaction radius [Å], excluding the displacement skin.
    virtual AccReal supported_cutoff() const = 0;

    /// @brief Total pair enumeration distance, including the displacement skin.
    /// Runtime also includes this distance in the communication halo.
    /// @return Enumeration radius [Å], at least supported_cutoff().
    virtual AccReal neighbor_reach() const = 0;

    /// @brief LAMMPS `units` resets `neighbor->skin`: the driver hands the
    ///        style's default skin [length] before it validates or builds.
    ///        A provider constructed with an explicit skin keeps it.
    virtual void apply_default_skin(double skin) = 0;

    /// @brief Validate the radii and return the allowed relative displacement.
    double validated_skin() const {
        const double cutoff = supported_cutoff();
        const double reach = neighbor_reach();
        if (!std::isfinite(cutoff) || cutoff < 0 || !std::isfinite(reach)
            || reach < cutoff)
            throw std::runtime_error(
                "neighbor provider requires finite nonnegative interaction "
                "cutoff "
                "and enumeration reach >= interaction cutoff");
        return reach - cutoff;
    }

    /// @brief Reject a consumer whose interaction radius consumes the skin.
    void validate_coverage(const NeighborConsumer& consumer,
                           const std::string& provider_name,
                           const std::string& consumer_name) const {
        (void)validated_skin();
        const double supported = supported_cutoff();
        const double required = consumer.neighbor_cutoff();
        if (!std::isfinite(required) || required < 0 || required > supported)
            throw std::runtime_error(
                "neighbor provider '" + provider_name
                + "' supports interaction cutoff " + std::to_string(supported)
                + ", but consumer '" + consumer_name + "' requires "
                + std::to_string(required)
                + "; the Verlet skin is not interaction coverage");
    }

    /// @brief LAMMPS `Neighbor::decide()` — one `nflag` for the whole step.
    ///
    /// Verlet calls this *before* exchange/borders so migrate, halo full
    /// rebuild, and the neighbor CSR rebuild share one gate. Default `true`
    /// (always reneighbor) is conservative for providers that have no
    /// Verlet cadence. `NeighborList` overrides with delay/every/check.
    ///
    /// @param rt Runtime context; the implementation writes nothing here
    ///        (the caller stores the bool on `rt.md.reneighbor`).
    /// @param frame Positions for the optional skin displacement test.
    /// @param md_cadence `true` under MD, where `delay` / `every` / `check`
    ///        apply. `false` for linear workflows: LAMMPS `Min::init` runs
    ///        every minimization pass at `every 1 delay 0 check yes`, so a
    ///        large line-search step cannot hide inside the delay window.
    /// @return `true` if this step should exchange + borders + rebuild.
    virtual bool decide_reneighbor(Context& rt, Frame& frame, bool md_cadence) {
        (void)rt;
        (void)frame;
        (void)md_cadence;
        return true;
    }
};

} // namespace cpu
} // namespace ATV
