/// @file cpu/driver/evaluate.h
/// @brief cpu::Evaluate — concrete one-pass evaluator driver.
///
/// `cpu::Evaluate` is the user-facing CPU single-point preset — a
/// `Driver<evaluate::Phase>` that runs the registered evaluator components
/// exactly once on the current `cpu::Frame` and returns. One pass IS the
/// termination: no time advance, no minimization loop.
///
/// Single-point energy / force / property workflows go here. Exposes
/// the same LAMMPS-shaped categorical setter API as `cpu::md::MD`
/// (minus the time-evolution and optimizer setters): set_evaluator,
/// set_evaluator, set_angle, set_dihedral, set_neighbor, set_observer,
/// set_record — all inherited from Driver.
///
/// Auto-adds ZeroForces and ZeroEnergy at `evaluate::Phase::prepare`
/// (orders -1000 / -999) so frame fields start clean before evaluators
/// write to them.
///
/// `add()` and `on(phase).add()` from the underlying `Driver<>`
/// remain public for user-defined hooks (e.g. a concrete SCF method
/// registered via `add(cpu::scf::RHF{})`).
///
/// Inheritance: Evaluate → Driver<evaluate::Phase> →
/// Driver<evaluate::Phase, cpu::device_tag>. There is no `EvaluateDriver` category in
/// between any more — a class whose whole contribution was "loop exactly once" is a
/// *budget of one*, not a taxonomy.
///
/// @code{.cpp}
///   // SCF energy + forces (default) — bind a concrete method
///   cpu::scf::RKS rks;
///   rks.basis = cpu::scf::BasisSet::STO3G;
///   cpu::Evaluate eval;
///   eval.on(evaluate::Phase::evaluate).add(std::move(rks));
///   eval.build();
///   eval.run(frame);
///
///   // Force-field force evaluation (LAMMPS-shaped)
///   cpu::Evaluate eval2;
///   eval2.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
///   eval2.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
///   eval2.build();
///   eval2.run(frame);
/// @endcode
#pragma once

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/pair_evaluator.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#ifdef ATV_HAS_MOLRS
#include "atomiverse/cpu/frame/molrs_exchange.h"
#endif

namespace ATV {
namespace cpu {

/// @brief One-pass evaluator preset — "compute these quantities at this geometry".
///
/// Single-point energy, force, density, dipole, charge. Categorical setters (set_evaluator,
/// set_evaluator, set_angle, set_dihedral, set_neighbor, set_observer, set_record) come from
/// Driver.
///
/// **No criterion.** One pass is not a termination policy — there is no loop to terminate.
class Evaluate : public Driver<ATV::evaluate::Phase> {
  public:
    Evaluate() {
        this->on(ATV::evaluate::Phase::prepare).add(cpu::ZeroForces{}, -1000);
        this->on(ATV::evaluate::Phase::prepare).add(cpu::ZeroEnergy{}, -999);
        // kStress accumulates from the pair/bonded virials just like the
        // forces do; a reused Evaluate driver would otherwise sum the virial
        // over every pass (mirrors MD's ZeroStress at pre_force −998).
        this->on(ATV::evaluate::Phase::prepare).add(cpu::ZeroStress{}, -998);
    }

    /// @brief Run a single evaluation pass through the compiled Plan.
    /// @param frame Host-resident system state, updated in place.
    void run(cpu::Frame& frame) { run(frame, /*step=*/0); }

    /// @brief One-pass evaluation at a caller-chosen step index.
    ///
    /// Same lifecycle as `run(frame)` (setup → one Plan pass → teardown). The
    /// step index is what time-dependent components see on `LaunchContext::step`
    /// (e.g. oscillating field phase). Prefer this over reaching into protected
    /// `setup_pass` / `step_once` — Frame remains the only user I/O token.
    ///
    /// @param frame Host-resident system state, updated in place.
    /// @param step 0-based step index published to components for this pass.
    void run(cpu::Frame& frame, int step) {
        setup_pass(frame);
        step_once(step);
        teardown_pass();
    }

    /// @brief User-facing entry — evaluate the system stored in a molrs.Frame.
    ///
    /// Materializes the molrs.Frame into a transient `cpu::Frame`, runs one evaluation
    /// pass on it, and writes the results back into the *same* molrs.Frame. The borrow
    /// window is strictly this one call.
    /// @param frame molrs.Frame handle (input geometry + output sink).
#ifdef ATV_HAS_MOLRS
    void run(molrs::FrameRef& frame) {
        cpu::Frame work = cpu::from_molrs(frame);
        run(work);
        cpu::to_molrs(work, frame);
    }
#endif

  protected:
    void install_defaults() override {
        const auto s = this->scan_pending_infra_();
        this->throw_if_missing_neighbor_list_(
            s, "Evaluate",
            "Add `eval.set_neighbor(cpu::md::NeighborList<>{cutoff, skin})`.");
    }

    void validate_components() const override {
        this->require_energy_or_force();
    }
};

} // namespace cpu
} // namespace ATV
