/// @file cuda/minimizer/minimize.cuh
/// @brief cuda::minimizer::Minimize — the CUDA energy-minimizer preset.
///
/// `Minimize` is the single user-facing CUDA minimizer: a
/// `cuda::CUDADriver<minimizer::Phase>` that owns a `run(frame, max_iter)` loop and a
/// build-time `validate_components()` contract (≥ 1 `Optimizer`-role component and ≥ 1
/// `Evaluator` carrying `Capability::Energy` or `Capability::Force`).
///
/// There is no `cuda::OptimizeDriver` category above it any more. "Stop when converged"
/// is not a *kind of driver*, it is a **policy**: `install_defaults()` attaches a
/// `cuda::ForceConverged` handler when the user attached none, and the user can replace
/// it — tighten the tolerances, add a NaN guard, compose both — without touching a
/// driver class or an optimizer:
///
/// @code{.cpp}
///   cuda::minimizer::Minimize min;
///   min.add(cuda::minimizer::LBFGS{});
///   min.add_handler(cuda::ForceConverged{/*ftol=*/1e-3});   // replaces the default
///   min.add_handler(cuda::StopOnNaN{});                     // …and composes with it
/// @endcode
///
/// Algorithms are hot-swappable components, not driver subclasses:
/// `cuda::minimizer::SteepestDescent`, `cuda::minimizer::CG` and
/// `cuda::minimizer::LBFGS` are Optimizer components plugged in via `add()`. Each of
/// them **measures and publishes** into `ctx.runtime.loop.metrics` (via the shared
/// `cuda::minimizer::ConvCheck`) and **decides nothing** — the handler decides. This
/// mirrors `cpu::minimizer::Minimize` exactly.
///
/// The constructor adds only infrastructure components. The phase-agnostic `Zero*`
/// initializers (`cuda/md/init.cuh`) are bound explicitly to `minimizer::Phase`.
///
/// @see cuda::CUDADriver for the loop primitives + the handler slot
/// @see cuda/driver/loop_handler.cuh for ForceConverged / StopOnNaN
/// @see cuda::minimizer::SteepestDescent / CG / LBFGS for the Optimizer components
/// @see minimizer::Phase for the minimizer phase sequence
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/driver/cuda_driver.cuh"
#include "atomiverse/cuda/driver/loop_handler.cuh"
#include "atomiverse/cuda/md/init.cuh"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cuda {
namespace minimizer {

/// @brief CUDA energy-minimization preset.
///
/// Binds the phase-agnostic `Zero*` infrastructure components to
/// `minimizer::Phase::force`:
/// - ZeroForces at force phase (order -1000) — zeros fx, fy, fz.
/// - ZeroEnergy at force phase (order -999) — zeros potential energy.
///
/// Usage:
/// @code{.cpp}
/// cuda::minimizer::Minimize min;
/// min.add(cuda::minimizer::SteepestDescent{0.01});
/// min.on(minimizer::Phase::force)
///    .add(cuda::NeighborList{3.0}, -10)
///    .add(cuda::PairLJCut{1.0, 1.0, 3.0});
/// min.build();          // install_defaults() → validate_components() → compile Plan
/// int iters = min.run(frame, 500);
/// bool ok = min.converged();          // the LOOP knows — not the optimizer
/// @endcode
///
/// Inheritance: Minimize → cuda::CUDADriver<minimizer::Phase> →
/// Driver<minimizer::Phase, cuda::device_tag>.
class Minimize : public cuda::CUDADriver<ATV::minimizer::Phase> {
  public:
    /// @brief Construct the minimizer, binding the phase-agnostic `Zero*`
    ///        infrastructure components to `minimizer::Phase::force`.
    Minimize() {
        on(ATV::minimizer::Phase::force).add(md::ZeroForces{}, -1000);
        on(ATV::minimizer::Phase::force).add(md::ZeroEnergy{}, -999);
    }

    /// @brief Iterate to convergence, or up to `max_iter` iterations.
    ///
    /// The Optimizer PUBLISHED its metrics during the step; the handlers judge them
    /// here. Neither `CG` nor `LBFGS` decides its own convergence any more — which is
    /// why a nested optimizer can no longer stop this loop.
    ///
    /// @param frame Host-resident system state; transferred H2D, minimized
    ///        on the device, retrieved D2H, updated in place.
    /// @param max_iter Iteration budget [dimensionless iteration count].
    /// @return Number of iterations actually executed (≤ `max_iter`).
    int run(cpu::Frame& frame, int max_iter) {
        setup_pass(frame);
        int iters = 0;
        for (int s = 0; s < max_iter; ++s) {
            this->mutable_outer_status().reset();
            step_once(s);
            ++iters;
            if (this->poll_handlers_(s) != Verdict::Continue)
                break;
        }
        teardown_pass(frame);
        return iters;
    }

  protected:
    /// @brief Optimizer + hot-path energy/force source.
    void validate_components() const override {
        bool has_optimizer = false;
        bool has_energy_or_force = false;
        for_each_pending([&](const cuda::Component& c) {
            // CUDA kinds not yet split — use Role flags.
            if (c.name().find("lbfgs") != std::string::npos
                || c.name().find("cg") != std::string::npos
                || c.name().find("sd") != std::string::npos
                || c.name().find("opt") != std::string::npos)
                has_optimizer = true;
            if (loop_required(c.role())
                && !has_role(c.role(), RoleFlag::FrameImmutable))
                has_energy_or_force = true;
        });
        // Prefer name-based optimizer detection until CUDA kinds land.
        for_each_pending([&](const cuda::Component& c) {
            if (c.name() == "lbfgs" || c.name() == "cg" || c.name() == "sd"
                || c.name() == "steepest_descent")
                has_optimizer = true;
        });
        if (!has_optimizer)
            throw_missing("Optimizer");
        if (!has_energy_or_force)
            throw_missing("hot-path energy/force Component");
    }

    /// @brief Attach the default termination policy, if the user attached none.
    ///
    /// A Minimize with no handler would never stop early — the optimizer publishes
    /// metrics and nobody judges them. So the PRESET supplies the policy it is a preset
    /// for. It contributes a default, not a semantics: textbook LAMMPS/ASE-ish
    /// tolerances, overridden wholesale with
    /// `min.add_handler(cuda::ForceConverged{ftol, …})` before `build()`. Same rule and
    /// same defaults as `cpu::minimizer::Minimize`.
    void install_defaults() override {
        if (!this->has_handler())
            this->add_handler(cuda::ForceConverged{});
    }
};

} // namespace minimizer
} // namespace cuda
} // namespace ATV
