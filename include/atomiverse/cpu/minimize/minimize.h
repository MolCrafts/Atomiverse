/// @file cpu/minimizer/minimize.h
/// @brief cpu::minimizer::Minimize — concrete energy-minimizer driver.
///
/// `Minimize` is the user-facing CPU geometry-optimization preset — a
/// `Driver<minimizer::Phase>` which attaches `cpu::ForceConverged` as a build-time default in `install_defaults()`.
/// It iterates an optimizer component (LBFGS / CG / FIRE), which publishes
/// force/displacement/energy metrics onto `ConvergenceMetrics`, until a criterion
/// judges them converged or `max_iter` is reached.
///
/// Exposes the same LAMMPS-shaped categorical setter API as
/// `cpu::md::MD` (minus the time-evolution setters): set_evaluator,
/// set_evaluator, set_angle, set_dihedral, set_neighbor, set_optimizer,
/// set_observer, set_record, set_thermo — all except set_optimizer
/// are inherited from Driver.
///
/// `add()` and `on(phase).add()` from the underlying `Driver<>`
/// remain public for user-defined hooks.
///
/// **MPI / domain decomposition.** Minimize mirrors `cpu::md::MD`'s
/// infrastructure auto-injection, folded into the single `force` phase
/// (the minimizer skeleton has no dedicated exchange/neighbor phases):
///
/// @code
///   force @ -1000  ZeroForces        (ctor)
///   force @  -999  ZeroEnergy        (ctor)
///   force @   -50  comm::Migrate     (install_defaults; re-homes owners,
///                                     single-rank no-op)
///   force @   -30  comm::HaloSend    (install_defaults, ATV_HAS_MPI +
///                                     pair evaluator; deduped)
///   force @   -10  NeighborList      (set_neighbor)
///   force @     0  pair / bonded evaluators
///   force @   +10  comm::HaloRecv    (install_defaults, gated as above)
///   force @   +20  md::mpi::Reduce   (install_defaults; makes
///                                     Frame metadata global *before* the
///                                     convergence_check phase reads them)
/// @endcode
///
/// The optimizer components are themselves DD-aware (owned-row loops,
/// globally-reduced inner products and convergence metrics — see
/// `lbfgs.h`). The recorders are DD-correct: `TrajectoryWriter` /
/// `FinalGeometry` gather the owner rows to rank 0
/// (`md::mpi::gather_to_root`, tag-ordered) before writing, and
/// `OptLog` reduces its metrics across ranks; all three write on
/// rank 0 only.
#pragma once

#include <memory>
#include <string>
#include <type_traits>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/asynchronous/snapshot_publisher.h"
#include "atomiverse/cpu/communication/atom_migration.h"
#include "atomiverse/cpu/communication/halo_exchange.h"
#include "atomiverse/cpu/communication/reduce.h"
#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/component/pair_evaluator.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/driver/driver.h"
#ifdef ATV_HAS_MOLRS
#include "atomiverse/cpu/frame/molrs_exchange.h"
#endif

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Energy-minimization driver — convergence-aware Optimize loop.
///
/// Auto-adds ZeroForces and ZeroEnergy at force phase. Wires the shared
/// `cpu::Publish` into `end_of_step @ +10` so async observers
/// see snapshots. `install_defaults()` injects the MPI infrastructure
/// documented in the file header (single-rank: structural no-ops).
///
/// `validate_components()` requires ≥ 1 `Optimizer` component (e.g. `LBFGS`,
/// `CG`) and ≥ 1 plugin that `provides_energy()` or `provides_force()`
/// (e.g. `SCF`).
///
/// Inheritance: Minimize → Driver<minimizer::Phase> →
/// Driver<minimizer::Phase, cpu::device_tag>. There is no `OptimizeDriver` category in
/// between any more: "stop when converged" is a `ForceConverged` criterion this preset
/// attaches in `install_defaults()`, and the user can replace it without touching a
/// driver class or an optimizer.
/// Driver provides the shared categorical setters and slot variables.
class Minimize : public Driver<ATV::minimizer::Phase> {
  public:
    /// @brief Completed run plus the still-live owner decomposition.
    struct DetachedRun {
        int iterations = 0;
        std::unique_ptr<cpu::md::mpi::DomainDecomposition> decomposition;
    };

    /// @brief Iterate to convergence, or up to `max_iter` iterations.
    ///
    /// Unlike MD (continuable: consecutive calls extend one trajectory), each `run()`
    /// here is a complete, self-contained optimization — `setup_pass` re-runs component
    /// `build()`, wiping the optimizer's curvature history, and `teardown_pass` runs
    /// before returning. That is deliberate: a converged geometry has no useful
    /// "continuation state", so re-running means re-optimizing (possibly after the caller
    /// perturbed the system) and stale L-BFGS history would be wrong to keep.
    ///
    /// @param frame Host-resident system state, updated in place.
    /// @param max_iter Iteration budget.
    /// @return Number of iterations actually executed (≤ `max_iter`).
    int run(cpu::Frame& frame, int max_iter) {
        return run_impl_(frame, max_iter, false).iterations;
    }

    /// @brief Optimize and transfer the final owner decomposition.
    ///
    /// Task orchestration uses this entry point to gather the final distributed
    /// Frame after component destruction. The decomposition is detached before
    /// Runtime shutdown, so its Cartesian communicator remains valid for gather.
    DetachedRun run_detached(cpu::Frame& frame, int max_iter) {
        return run_impl_(frame, max_iter, true);
    }

    /// @brief User-facing entry — optimize the system stored in a molrs.Frame.
    /// @param frame molrs.Frame handle (input geometry + output sink).
    /// @param max_iter Iteration budget.
    /// @return Number of iterations actually executed (≤ `max_iter`).
#ifdef ATV_HAS_MOLRS
    int run(molrs::FrameRef& frame, int max_iter) {
        cpu::Frame work = cpu::from_molrs(frame);
        const int iters = run(work, max_iter);
        cpu::to_molrs(work, frame);
        return iters;
    }
#endif

    Minimize() {
        // Phase-agnostic backend-root Zero*, bound to minimizer phases.
        // kStress is accumulated by the pair/bonded virials exactly like the
        // forces are (`PairLJCut` does `+=`), so it needs the same per-iteration
        // clean slate — without it a cell relax sums the virial over every
        // minimizer iteration and reports a pressure that only grows.
        this->on(ATV::minimizer::Phase::force)
            .add(cpu::ZeroForces{}, -1000)
            .add(cpu::ZeroEnergy{}, -999)
            .add(cpu::ZeroStress{}, -998);
        this->on(ATV::minimizer::Phase::end_of_step).add(cpu::Publish{}, 10);
    }

    // ─── Minimizer-specific categorical setters ─────────────────────
    //
    // set_evaluator, set_evaluator, set_angle, set_dihedral, set_neighbor,
    // set_observer, set_record, set_thermo are inherited from Driver.

    /// @brief Configure the optimization algorithm (LAMMPS `min_style`).
    template <typename O>
        requires(!std::is_lvalue_reference_v<O>)
    Minimize& set_optimizer(O&& o) {
        this->set_singular_default_owned_(std::forward<O>(o), optimizer_set_,
                                          "set_optimizer");
        return *this;
    }

    template <typename O>
    Minimize& set_optimizer_ref(O& o) {
        this->set_singular_default_ref_(o, optimizer_set_, "set_optimizer_ref");
        return *this;
    }

    template <typename O>
    Minimize& set_optimizer(O&) = delete;

    /// @brief Owner-migration cadence in optimizer iterations (default 10,
    /// matching `HaloSend`'s full-rebuild cadence). Each actual migration
    /// discards the optimizer's curvature history (row indices reshuffle),
    /// so raise this when minimizing systems that barely drift; the
    /// exchange itself is skipped whenever no rank has a strayed atom.
    /// Call before `build()`.
    Minimize& set_migrate_every(int every) {
        migrate_every_ = every < 1 ? 1 : every;
        return *this;
    }

  protected:
    void validate_components() const override {
        this->require_optimizer();
        this->require_energy_or_force();
    }

    void install_defaults() override {
        // A Minimize with no criterion would never stop early — the optimizer publishes
        // metrics and nobody judges them. So the PRESET supplies the policy it is a
        // preset for. This is what "MD-style pipelines become a default-constructed
        // driver + criterion" means in practice: the class contributes a default, not a
        // category.
        //
        // Textbook LAMMPS/ASE-ish defaults; override wholesale with
        // `opt.add_criterion(cpu::ForceConverged{ftol, ...})` before build().
        if (!this->has_criterion())
            this->add_criterion(cpu::ForceConverged{});

        const auto s = this->scan_pending_infra_();

        // Owner migration + global scalar reduction, explicitly bound to
        // minimizer phases (their default_bindings() are md::Phase values
        // and must not be used here). Both are single-rank no-ops. Reduce
        // sits after HaloRecv (+10) and before convergence_check so the
        // optimizer reads globally-summed energy.
        this->on(ATV::minimizer::Phase::force)
            .add(comm::Migrate{migrate_every_}, -50);
        this->on(ATV::minimizer::Phase::force).add(md::mpi::Reduce{}, +20);

#ifdef ATV_HAS_MPI
        if (s.neighbor_consumer_cutoff) {
            // Same 10% skin rule as MD::install_defaults — ghosts must stay
            // registered as atoms drift between the every-10-step full halo
            // rebuilds. The minimizer's step cap (0.1 Å) keeps per-step
            // drift well inside this band. Explicit halo registrations
            // suppress the auto-inject: the reverse exchange must run
            // exactly once (see halo_exchange.h).
            const double halo_cut =
                1.1 * static_cast<double>(*s.neighbor_consumer_cutoff);
            if (!s.has_halo_send)
                this->on(ATV::minimizer::Phase::force)
                    .add(comm::HaloSend{halo_cut}, -30);
            if (!s.has_halo_recv)
                this->on(ATV::minimizer::Phase::force)
                    .add(comm::HaloRecv{}, +10);
        }
#endif

        this->throw_if_missing_neighbor_list_(
            s, "Minimize",
            "Add `opt.set_neighbor(cpu::md::NeighborList<>{cutoff, skin})`.");
    }

  private:
    DetachedRun run_impl_(cpu::Frame& frame, int max_iter, bool detach_dd) {
        setup_pass(frame);
        // Publish the re-entrant force re-eval handle so an Optimizer component can
        // evaluate trial geometries during the line_search phase. Replays the force
        // phase (zeroing → evaluator → reduce) on the frame's current geometry.
        // Captures `this`; valid for the run.
        this->backend_->context().opt.reeval_forces =
            [this](bool /*need_grad*/) {
                this->backend_->run_stage(Traits::eval_phase, /*step_idx=*/0);
            };
        int iters = 0;
        for (int s = 0; s < max_iter; ++s) {
            mutable_outer_status().reset();
            step_once(s);
            ++iters;
            // The optimizer publishes metrics; criteria alone judge convergence.
            if (poll_criteria_(s) != Verdict::Continue)
                break;
        }

        DetachedRun result;
        result.iterations = iters;
        if (detach_dd)
            result.decomposition = teardown_pass_releasing_dd();
        else
            teardown_pass();
        return result;
    }

    std::optional<std::string> optimizer_set_;
    int migrate_every_ = 10;
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
