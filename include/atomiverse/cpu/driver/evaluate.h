/// @file cpu/driver/evaluate.h
/// @brief cpu::Evaluate — concrete one-pass evaluator driver.
///
/// `cpu::Evaluate` is the user-facing CPU single-point preset — a
/// `CPUDriver<evaluate::Phase>` that runs the registered evaluator components
/// exactly once on the current `cpu::Frame` and returns. One pass IS the
/// termination: no time advance, no minimization loop.
///
/// Single-point energy / force / property workflows go here. Exposes
/// the same LAMMPS-shaped categorical setter API as `cpu::md::MD`
/// (minus the time-evolution and optimizer setters): add_potential,
/// add_potential, set_angle, set_dihedral, set_neighbor, set_observer,
/// set_record — all inherited from CPUDriver.
///
/// Auto-adds ZeroForces and ZeroEnergy at `evaluate::Phase::prepare`
/// (orders -1000 / -999) so frame fields start clean before evaluators
/// write to them.
///
/// `add()` and `on(phase).add()` from the underlying `Driver<>`
/// remain public for user-defined hooks (e.g. a concrete SCF method
/// registered via `add(cpu::scf::RHF{})`).
///
/// Inheritance: Evaluate → CPUDriver<evaluate::Phase> →
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
///   eval2.add_potential(cpu::PairLJCut{eps, sigma, cutoff});
///   eval2.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
///   eval2.build();
///   eval2.run(frame);
/// @endcode
#pragma once

#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/pair_evaluator.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/driver/cpu_driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/materialize.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {

/// @brief One-pass evaluator preset — "compute these quantities at this geometry".
///
/// Single-point energy, force, density, dipole, charge. Categorical setters (add_potential,
/// add_potential, set_angle, set_dihedral, set_neighbor, set_observer, set_record) come from
/// CPUDriver.
///
/// **No criterion.** One pass is not a termination policy — there is no loop to terminate.
class Evaluate : public CPUDriver<ATV::evaluate::Phase> {
  public:
    Evaluate() {
        this->on(ATV::evaluate::Phase::prepare).add(cpu::ZeroForces{}, -1000);
        this->on(ATV::evaluate::Phase::prepare).add(cpu::ZeroEnergy{}, -999);
    }

    /// @brief Run a single evaluation pass through the compiled Plan.
    /// @param frame Host-resident system state, updated in place.
    void run(cpu::Frame& frame) {
        setup_pass(frame);
        step_once(0);
        teardown_pass();
    }

    /// @brief User-facing entry — evaluate the system stored in a molrs.Frame.
    ///
    /// Materializes the molrs.Frame into a transient `cpu::Frame`, runs one evaluation
    /// pass on it, and writes the results back into the *same* molrs.Frame. The borrow
    /// window is strictly this one call.
    /// @param frame molrs.Frame handle (input geometry + output sink).
    void run(molrs::FrameRef& frame) {
        cpu::Frame work = cpu::materialize(frame);
        run(work);
        cpu::write_back(work, frame);
    }

  protected:
    void install_defaults() override {
        const auto s = this->scan_pending_infra_();
        this->throw_if_missing_neighbor_list_(
            s, "Evaluate",
            "Add `eval.set_neighbor(cpu::md::NeighborList<>{cutoff, skin})`.");
    }

    /// @brief At least one energy/force kind (Potential, Bias, HF, …).
    ///        Component nodes and nested method-drivers (pure HF tag) both count.
    void validate_components() const override {
        bool found = false;
        for_each_pending_kind(
            [&](const cpu::Component& c) {
                if (dynamic_cast<const Potential*>(&c)
                    || dynamic_cast<const Bias*>(&c)
                    || dynamic_cast<const PathCoupling*>(&c)
                    || dynamic_cast<const HF*>(&c)
                    || dynamic_cast<const Semiempirical*>(&c)
                    || dynamic_cast<const PlaneWave*>(&c)
                    || dynamic_cast<const GPW*>(&c))
                    found = true;
            },
            [&](const cpu::Driver* d) {
                if (dynamic_cast<const HF*>(d))
                    found = true;
            });
        if (!found)
            throw_missing(
                "Potential / Bias / HF / Semiempirical / PlaneWave / GPW");
    }
};

} // namespace cpu
} // namespace ATV
