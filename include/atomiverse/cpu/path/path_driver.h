/// @file cpu/driver/path_driver.h
/// @brief cpu::PathDriver — the coupled-replica preset: N replicas, all must converge.
///
/// "N coupled replicas" is a **second axis**, orthogonal to termination — see
/// `cpu/replica/replica_set.h`, which is where the replica machinery lives so that a
/// future budget-terminating replica preset (REMD) reuses it instead of cloning it.
///
/// ## Termination
///
/// The driver publishes `ReplicaSet::converged_fraction()` onto
/// `ConvergenceMetrics::replicas_converged`, and the `cpu::AllReplicasConverged`
/// criterion this preset attaches applies the AND (converged at exactly 1.0). One
/// unconverged image keeps the whole chain running. The rule is a criterion component,
/// so a user can compose or replace it (`neb.add_criterion(StopOnNaN{})`) without
/// touching a driver class.
///
/// ## The loop
///
/// @code
///   expand one Frame -> N replicas (ReplicaExpansion)
///   repeat:
///     for each replica i:            <- force is PER-replica
///        point the backend at replica i's Frame; run the `force` phase
///     run `couple`                   <- band-wide: sees all replicas via ctx.path
///     run `search_direction` .. `end_of_step`
///     stop when every replica reports converged
///   collapse N replicas -> the caller's one Frame
/// @endcode
///
/// ## Why ONE backend and not N
///
/// The spec for this probe called for a Backend (and therefore a Context, and therefore
/// an `IterationStatus`) **per replica**, which would have given collective termination for
/// free. It is not implementable, and finding that out is one of this probe's results:
/// **components cannot be cloned.** `PhaseBuilder::add()` *moves* a component into
/// `owned_components_`, and there is no `clone()` anywhere in the tree — so N backends
/// would need N independently-constructed component sets, which the user would have to
/// supply through a factory callback (against both the OOP style rule and any hope of a
/// decent API), or which would need a `clone()` virtual bolted onto `Component` —
/// a dead interface for every component that is not in a band.
///
/// So: one backend, one plan, and the band reached through `ctx.path`. The `force` phase
/// is replayed once per replica by re-pointing the backend's frame; the band-wide phases
/// run once and see everything. Per-replica convergence lives on `ReplicaSet`, not on N
/// copies of `IterationStatus`; `PathDriver` publishes it as a fraction onto
/// `ConvergenceMetrics` and the criterion applies the AND.
///
/// Known ceiling of the one-backend route: force-phase components that cache per-frame
/// state (a neighbor list's bins) get invalidated on every replica switch and rebuild N
/// times per iteration. Correct, but O(N) rebuilds. A band of Müller-Brown particles has
/// no neighbor list at all; a band of a real solid would want a per-replica neighbor
/// cache, which is the point at which a component *would* need to be replica-aware.
#pragma once

#include <stdexcept>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/replica/replica_set.h"

namespace ATV {
namespace cpu {

class PathDriver : public Driver<ATV::path::Phase> {
  public:
    /// @brief The coupled-replica preset. Attaches its default termination criterion.
    ///
    /// `AllReplicasConverged` reads `ConvergenceMetrics::replicas_converged` (the
    /// fraction of the band that has met its criterion) and stops the loop only at 1 —
    /// the AND over every image that this driver used to spell out by hand inside its
    /// own `run()`.
    ///
    /// That hand-written `if` is exactly the shape probe 3 was forced into, and exactly
    /// what a category makes unavoidable: the termination rule could not be named, so it
    /// could not be replaced. Now it can — `neb.add_criterion(StopOnNaN{})` composes
    /// with it, and a user who wants "stop when the barrier estimate settles" writes
    /// that criterion instead of a seventh driver class.
    ///
    /// It also binds the driver-owned accumulator zeroing at the head of the
    /// `force` phase (orders −1000 / −999, mirroring `md::MD`). The `force`
    /// stage is replayed once per replica, so the pair runs once per image —
    /// exactly the right scope: an accumulating (`+=`) force source would
    /// otherwise carry image `i`'s forces into iteration `s+1` and grow the
    /// band's forces linearly with the iteration count. `NebCoupling` zeroes
    /// only the frozen endpoints, so nothing else clears the interior images.
    PathDriver() {
        this->add_criterion(cpu::AllReplicasConverged{});
        this->on(ATV::path::Phase::force)
            .add(cpu::ZeroForces{}, -1000)
            .add(cpu::ZeroEnergy{}, -999);
    }

    /// @brief Advance the band until every replica converges, or `max_iter` runs out.
    ///
    /// @param frame Initial state. Expanded into the band on the way in; **overwritten**
    ///        with the collapsed result on the way out (for CI-NEB: the saddle-point
    ///        geometry). Frame in, Frame out — the facade currency is unchanged.
    /// @param expansion The strategy that generates the band. Borrowed for the call.
    /// @param max_iter Iteration budget for the whole chain.
    /// @return Iterations actually executed.
    int run(cpu::Frame& frame, const ReplicaExpansion& expansion,
            int max_iter) {
        ReplicaSet band(frame, expansion);

        // setup_pass binds the backend to replica 0's frame. Every replica has the same
        // atom count and topology (they are expansions of ONE frame), so components size
        // their buffers once and stay valid as the active frame is swapped underneath.
        setup_pass(band.frame(0));

        auto& ctx = this->backend_->context();
        ctx.path.replicas = &band;

        int iters = 0;
        for (int s = 0; s < max_iter; ++s) {
            mutable_outer_status().reset();

            // ── Per-replica: force ────────────────────────────────────────────
            for (int i = 0; i < band.size(); ++i) {
                ctx.path.active = i;
                this->backend_->set_frame(band.frame(i));
                this->backend_->run_stage(ATV::path::Phase::force, s);
            }
            ctx.path.active = -1;

            // ── Band-wide: couple → direction → line search → convergence → eos ──
            // These see every replica through ctx.path.replicas. The active frame is
            // left pointing at the last replica and is deliberately not meaningful
            // here; a band-wide component that reads ctx.frame is reading one arbitrary
            // image, which is a bug in that component.
            for (const auto p :
                 {ATV::path::Phase::couple, ATV::path::Phase::search_direction,
                  ATV::path::Phase::line_search,
                  ATV::path::Phase::convergence_check,
                  ATV::path::Phase::end_of_step}) {
                this->backend_->run_stage(p, s);
            }

            ++iters;

            // PUBLISH what the band measured; let the criterion DECIDE. The driver used
            // to write `converged = true` itself the moment every replica had settled —
            // the same "the algorithm judges itself" mistake LBFGS made, and the reason
            // the replica table had to exist in the first place.
            metrics().replicas_converged = band.converged_fraction();

            if (poll_criteria_(s) != Verdict::Continue)
                break;
        }

        band.collapse_into(frame, expansion);
        ctx.path.replicas = nullptr;
        teardown_pass();
        return iters;
    }

  protected:
    /// A Path run needs something to compute forces on each replica and something to
    /// move them. The coupling component is NOT required here: an uncoupled band is
    /// just N independent minimizations, which is a legitimate (if pointless) thing to
    /// run, and refusing it would mean this driver knew what NEB is. Nested
    /// method-drivers count as HF force sources via cross-cast.
    void validate_components() const override {
        bool has_optimizer = false;
        bool has_force = false;
        for_each_pending_kind(
            [&](const cpu::Component& c) {
                if (dynamic_cast<const Optimizer*>(&c))
                    has_optimizer = true;
                if (dynamic_cast<const Potential*>(&c)
                    || dynamic_cast<const Bias*>(&c)
                    || dynamic_cast<const PathCoupling*>(&c)
                    || dynamic_cast<const HF*>(&c)
                    || dynamic_cast<const Semiempirical*>(&c)
                    || dynamic_cast<const PlaneWave*>(&c)
                    || dynamic_cast<const GPW*>(&c))
                    has_force = true;
            },
            [&](const cpu::DriverNode* d) {
                if (dynamic_cast<const HF*>(d))
                    has_force = true;
            });
        if (!has_force)
            throw_missing(
                "Potential / Bias / HF / Semiempirical / PlaneWave / GPW");
        if (!has_optimizer)
            throw_missing("Optimizer");
    }
};

} // namespace cpu
} // namespace ATV
