/// @file cuda/driver/evaluate.cuh
/// @brief cuda::Evaluate — one-pass CUDA evaluator preset.
///
/// `cuda::Evaluate` runs the registered evaluator components exactly once on the current
/// `cpu::Frame` (transferred H2D, evaluated on the device, retrieved D2H) and returns —
/// no time advance, no minimization loop. "Compute one or more quantities at this
/// geometry": single-point energy, force, density, dipole, charge.
///
/// Single-point CUDA SCF energy / density workflows go here:
/// `cuda::Evaluate + cuda::scf::SCF`.
///
/// Validation rule (build-time): at least one hot-path energy/force
/// Component (Role flags; no Capability bitset).
///
/// **No handler.** One pass is not a termination policy — there is no loop to terminate.
///
/// Inheritance: Evaluate → cuda::CUDADriver<evaluate::Phase> →
/// Driver<evaluate::Phase, cuda::device_tag>. There is no `cuda::EvaluateDriver`
/// category in between any more — a class whose whole contribution was "loop exactly
/// once" is a *budget of one*, not a taxonomy.
///
/// @code{.cpp}
///   cuda::Evaluate eval;
///   eval.on(evaluate::Phase::evaluate).add(cuda::scf::SCF{integrals, tol, max_iter});
///   eval.build();
///   eval.run(frame);            // frame.scalars.energy populated
/// @endcode
///
/// @see cuda::single_point for the free-function single-evaluator helper
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/driver/cuda_driver.cuh"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cuda {

/// @brief One-pass CUDA evaluator preset.
class Evaluate : public CUDADriver<ATV::evaluate::Phase> {
  public:
    /// @brief Run a single evaluation pass through the compiled Plan.
    /// @param frame Host-resident system state; transferred H2D, evaluated
    ///        on the device, retrieved D2H, updated in place.
    void run(cpu::Frame& frame) {
        setup_pass(frame);
        step_once(0);
        teardown_pass(frame);
    }

  protected:
    /// @brief Require at least one hot-physics energy/force source.
    void validate_components() const override {
        bool found = false;
        for_each_pending([&](const cuda::Component& c) {
            if (loop_required(c.role())
                && !has_role(c.role(), RoleFlag::FrameImmutable)
                && !has_role(c.role(), RoleFlag::ExternalIO))
                found = true;
        });
        if (!found)
            throw_missing("hot-path energy/force Component");
    }
};

} // namespace cuda
} // namespace ATV
