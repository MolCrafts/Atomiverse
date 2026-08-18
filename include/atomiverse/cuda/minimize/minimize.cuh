/// @file cuda/minimizer/minimize.cuh
/// @brief cuda::minimizer::Minimize — the CUDA energy-minimizer preset.
///
/// `Minimize` is the single user-facing CUDA minimizer: a
/// `cuda::Driver<minimizer::Phase>` that owns a `run(frame, max_iter)` loop and a
/// build-time `validate_components()` contract (≥ 1 `Optimizer` and ≥ 1
/// plugin that `provides_energy()` or `provides_force()`).
///
/// There is no `cuda::OptimizeDriver` category above it any more. "Stop when converged"
/// is not a *kind of driver*, it is a **policy**: `install_defaults()` attaches a
/// `cuda::ForceConverged` handler when the user attached none, and the user can replace
/// it — tighten the tolerances, add a NaN guard, compose both — without touching a
/// driver class or an optimizer:
///
/// @code{.cpp}
///   cuda::minimizer::Minimize min;
///   min.set_optimizer(cuda::minimizer::LBFGS{});
///   min.add_criterion(cuda::ForceConverged{/*ftol=*/1e-3});   // replaces the default
///   min.add_criterion(cuda::StopOnNaN{});                     // …and composes with it
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
/// @see cuda::Driver for the loop primitives + the handler slot
/// @see cuda/driver/criterion.cuh for ForceConverged / StopOnNaN
/// @see cuda::minimizer::SteepestDescent / CG / LBFGS for the Optimizer components
/// @see minimizer::Phase for the minimizer phase sequence
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/criterion.cuh"
#include "atomiverse/cuda/driver/driver.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"

namespace ATV {
namespace cuda {
namespace minimizer {

/// @brief CUDA energy-minimization preset.
///
/// Binds the phase-agnostic `Zero*` infrastructure components to
/// `minimizer::Phase::force`:
/// - ZeroForces at force phase (order -1000) — zeros fx, fy, fz.
/// - ZeroEnergy at force phase (order -999) — zeros potential energy.
/// - ZeroStress at force phase (order -998) — zeros the virial tensor.
///
/// Usage:
/// @code{.cpp}
/// cuda::minimizer::Minimize min;
/// min.set_optimizer(cuda::minimizer::SteepestDescent{0.01});
/// min.set_neighbor(cuda::NeighborList{3.0});
/// min.set_evaluator(cuda::PairLJCut{1.0, 1.0, 3.0});
/// min.build();          // install_defaults() → validate_components() → compile Plan
/// int iters = min.run(frame, 500);
/// bool ok = min.converged();          // the LOOP knows — not the optimizer
/// @endcode
///
/// Inheritance: Minimize → cuda::Driver<minimizer::Phase> →
/// Driver<minimizer::Phase, cuda::device_tag>.
class Minimize : public cuda::Driver<ATV::minimizer::Phase> {
  public:
    /// @brief Construct the minimizer, binding the phase-agnostic `Zero*`
    ///        infrastructure components to `minimizer::Phase::force`.
    Minimize() {
        on(ATV::minimizer::Phase::force).add(md::ZeroForces{}, -1000);
        on(ATV::minimizer::Phase::force).add(md::ZeroEnergy{}, -999);
        on(ATV::minimizer::Phase::force).add(md::ZeroStress{}, -998);
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

    /// set_evaluator / set_neighbor / set_observer / set_record come from
    /// cuda::Driver.

    /// @brief Configure the optimization algorithm (LAMMPS `min_style`).
    template <typename O,
              std::enable_if_t<!std::is_lvalue_reference_v<O>, int> = 0>
    Minimize& set_optimizer(O&& o) {
        const std::string n = o.name();
        if (optimizer_set_) {
            if (*optimizer_set_ == n)
                return *this;
            throw std::invalid_argument(
                "set_optimizer already configured with '" + *optimizer_set_
                + "', rejected re-call with '" + n + "'");
        }
        optimizer_set_ = n;
        this->bind_defaults(std::forward<O>(o));
        return *this;
    }
    template <typename O>
    Minimize& set_optimizer_ref(O& o) {
        const std::string n = o.name();
        if (optimizer_set_) {
            if (*optimizer_set_ == n)
                return *this;
            throw std::invalid_argument(
                "set_optimizer already configured with '" + *optimizer_set_
                + "', rejected re-call with '" + n + "'");
        }
        optimizer_set_ = n;
        this->bind_defaults_ref(o);
        return *this;
    }
    template <typename O>
    Minimize& set_optimizer(O&) = delete;

  protected:
    /// @brief Optimizer + energy or force plugin.
    void validate_components() const override {
        this->require_optimizer();
        this->require_energy_or_force();
    }

    /// @brief Attach the default termination policy, if the user attached none.
    ///
    /// A Minimize with no handler would never stop early — the optimizer publishes
    /// metrics and nobody judges them. So the PRESET supplies the policy it is a preset
    /// for. It contributes a default, not a semantics: textbook LAMMPS/ASE-ish
    /// tolerances, overridden wholesale with
    /// `min.add_criterion(cuda::ForceConverged{ftol, …})` before `build()`. Same rule and
    /// same defaults as `cpu::minimizer::Minimize`.
    void install_defaults() override {
        if (!this->has_handler())
            this->add_criterion(cuda::ForceConverged{});
        const auto s = this->scan_pending_infra_();
        this->throw_if_missing_neighbor_list_(
            s, "cuda::minimizer::Minimize",
            "Add `min.set_neighbor(cuda::NeighborList{cutoff})`.");
    }

    void clear_singular_on_unfix_(std::string_view name_or_id) override {
        if (optimizer_set_ && *optimizer_set_ == name_or_id)
            optimizer_set_.reset();
    }

  private:
    std::optional<std::string> optimizer_set_;
};

} // namespace minimizer
} // namespace cuda
} // namespace ATV
