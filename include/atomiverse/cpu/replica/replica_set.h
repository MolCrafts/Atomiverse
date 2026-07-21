/// @file cpu/replica/replica_set.h
/// @brief ReplicaSet — N coupled replicas + their collective iteration status.
///
/// **This is infrastructure, not a Driver.** The distinction is the whole finding of
/// probe #3, and it is worth stating plainly, because the Driver taxonomy as written in
/// CLAUDE.md cannot express it:
///
/// > "A Driver category is decided by **termination semantics**, never by algorithm or
/// > method."
///
/// That is one axis. "N coupled replicas" is a **second, orthogonal** axis:
///
/// |                  | terminate on convergence   | terminate on a budget          |
/// |------------------|----------------------------|--------------------------------|
/// | **one Frame**    | `cpu::minimizer::Minimize` | `cpu::md::MD` / `cpu::mc::MC`  |
/// | **N replicas**   | **`cpu::PathDriver`** (NEB)| REMD / PIMD / umbrella (later) |
///
/// Read one-dimensionally, REMD and PIMD would each eventually demand a driver class of
/// their own, and each would re-implement ~90 % of the replica machinery that NEB already
/// needed. So the replica machinery lives HERE, reusable, and `PathDriver` contributes
/// exactly one thing on top of it: it publishes `converged_fraction()` onto
/// `ConvergenceMetrics`, where `cpu::AllReplicasConverged` judges it.
///
/// A budget-terminating replica preset (REMD) would reuse this class unchanged and
/// contribute *its* one thing (the exchange attempt). That is the test of whether this
/// split was drawn in the right place.
#pragma once

#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/replica/replica_expansion.h"

namespace ATV {
namespace cpu {

/// @brief The band: N replicas expanded from one Frame, plus their collective status.
class ReplicaSet {
  public:
    /// @brief Expand `initial` through `expansion`.
    ///
    /// Constructed from `(Frame, ReplicaExpansion&)` — deliberately **not** from a
    /// user-supplied `std::vector<Frame>`. Handing the driver N frames would make the
    /// user responsible for building the band, and would fight the facade signature
    /// (`run()` takes ONE Frame). Replicas are *generated*, always.
    ///
    /// @param initial The Frame passed to `Driver::run()`.
    /// @param expansion The strategy; borrowed for the duration of this call only.
    ReplicaSet(const cpu::Frame& initial, const ReplicaExpansion& expansion)
        : replicas_(expansion.expand(initial)),
          converged_(replicas_.size(), false) {}

    /// @brief Number of replicas.
    int size() const { return static_cast<int>(replicas_.size()); }

    /// @brief Mutable access to replica `i`'s Frame (the driver sets it as the
    ///        backend's active frame; coupling components read their neighbours).
    cpu::Frame& frame(int i) { return replicas_[i].frame; }
    const cpu::Frame& frame(int i) const { return replicas_[i].frame; }

    /// @brief The expansion-assigned scalar for replica `i` (NEB: path coordinate;
    ///        REMD: temperature; PIMD: bead index). @see cpu::Replica::parameter
    double parameter(int i) const { return replicas_[i].parameter; }

    /// @brief Record whether replica `i` has met its stopping criterion.
    ///
    /// Per-replica, not global — because a chain is not converged when its *easiest*
    /// image is. When an optimizer could write convergence straight onto a single shared
    /// status, the first image to settle would have stopped the whole band.
    /// @param i Replica index.
    /// @param c True if replica `i` has met its stopping criterion.
    void set_converged(int i, bool c) { converged_[i] = c; }

    /// @brief Fraction of the band that has converged, in [0,1]. Empty band → 0.
    ///
    /// Published each iteration onto `ConvergenceMetrics::replicas_converged` so that a
    /// *criterion* applies the AND (the band is converged at exactly 1.0) instead of the
    /// driver hardcoding it. AND, not OR, and not "the climbing image converged": a band
    /// whose saddle image is converged but whose neighbours are still sliding has not
    /// found a path — the tangent that defines the saddle is computed FROM those
    /// neighbours. A metric is something you measure; a verdict is something you decide,
    /// and those are different jobs. @see cpu::AllReplicasConverged
    /// @return Converged fraction in [0,1] (dimensionless).
    double converged_fraction() const {
        if (converged_.empty())
            return 0.0;
        int n = 0;
        for (const bool c : converged_)
            if (c)
                ++n;
        return static_cast<double>(n) / static_cast<double>(converged_.size());
    }

    /// @brief Fold the band back into the caller's single Frame. Frame in, Frame out.
    void collapse_into(cpu::Frame& out,
                       const ReplicaExpansion& expansion) const {
        expansion.collapse(replicas_, out);
    }

  private:
    std::vector<Replica> replicas_;
    std::vector<bool> converged_;
};

} // namespace cpu
} // namespace ATV
