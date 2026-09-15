/// @file cuda/evaluate/evaluate.cuh
/// @brief cuda::Evaluate — one-pass CUDA evaluator preset.
///
/// `cuda::Evaluate` runs the registered evaluator components exactly once on the current
/// `cpu::Frame` (transferred H2D, evaluated on the device, retrieved D2H) and returns —
/// no time advance, no minimization loop. "Compute one or more quantities at this
/// geometry": single-point energy, force, density, dipole, charge.
///
/// Single-point CUDA SCF energy / density workflows go here:
/// `cuda::Evaluate + cuda::scf::RHF` (preferred) or the synthetic
/// `cuda::scf::SCF` test path.
///
/// Validation rule (build-time): at least one plugin that
/// `provides_energy()` or `provides_force()`.
///
/// **No handler.** One pass is not a termination policy — there is no loop to terminate.
///
/// Inheritance: Evaluate → cuda::Driver<evaluate::Phase> →
/// Driver<evaluate::Phase, cuda::device_tag>. There is no `cuda::EvaluateDriver`
/// category in between any more — a class whose whole contribution was "loop exactly
/// once" is a *budget of one*, not a taxonomy.
///
/// @code{.cpp}
///   cuda::Evaluate eval;
///   eval.set_evaluator(cuda::scf::RHF{cpu::scf::BasisSet::STO3G});
///   eval.build();
///   eval.run(frame);            // frame.meta.get(kEnergy) [eV]
/// @endcode
///
/// @see cuda::single_point for the free-function single-evaluator helper
#pragma once

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"

namespace ATV {
namespace cuda {

/// @brief One-pass CUDA evaluator preset.
///
/// Categorical setters (`set_evaluator` / `set_neighbor` / `set_observer` /
/// `set_record`) come from `cuda::Driver`. Constructor binds ZeroForces /
/// ZeroEnergy / ZeroStress at `prepare`, matching `cpu::Evaluate`.
class Evaluate : public Driver<ATV::evaluate::Phase> {
  public:
    Evaluate() {
        this->on(ATV::evaluate::Phase::prepare).add(md::ZeroForces{}, -1000);
        this->on(ATV::evaluate::Phase::prepare).add(md::ZeroEnergy{}, -999);
        this->on(ATV::evaluate::Phase::prepare).add(md::ZeroStress{}, -998);
    }

    /// @brief Run a single evaluation pass through the compiled Plan.
    /// @param frame Host-resident system state; transferred H2D, evaluated
    ///        on the device, retrieved D2H, updated in place.
    void run(cpu::Frame& frame) {
        setup_pass(frame);
        step_once(0);
        teardown_pass(frame);
    }

  protected:
    void install_defaults() override {
        const auto s = this->scan_pending_infra_();
        this->throw_if_missing_neighbor_list_(
            s, "cuda::Evaluate",
            "Add `eval.set_neighbor(cuda::NeighborList{cutoff})`.");
    }

    /// @brief Require at least one energy or force plugin.
    void validate_components() const override {
        this->require_energy_or_force();
    }
};

} // namespace cuda
} // namespace ATV
