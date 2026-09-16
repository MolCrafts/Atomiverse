/// @file cpu/semiempirical/am1_evaluator.h
/// @brief AM1 Evaluator component for Atomiverse drivers.
#pragma once

#include <string>
#include <utility>

#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief AM1 semiempirical Evaluator (owns concrete `AM1` method).
///
/// The Component face of `AM1` — an energy/force source a Driver can bind.
/// Each `launch()` runs one SCF solve at the frame's current geometry and
/// publishes total energy [eV], Mulliken charges [e] and, when gradients are
/// enabled and the phase asks for them, nuclear forces [eV/Å]. Bind it under
/// an `MD` force phase and the run is Born-Oppenheimer dynamics: the electrons
/// are re-converged at every geometry the nuclei reach.
class AM1Evaluator : public cpu::Semiempirical {
  public:
    /// @brief Default AM1 settings (neutral singlet; see `AM1`'s members).
    AM1Evaluator() = default;
    /// @brief Own a concrete AM1 method (parameters already on the method).
    explicit AM1Evaluator(AM1 method) : solver_(std::move(method)) {}
    /// @brief Construct method with SCF parameters (same args as `AM1`).
    AM1Evaluator(int charge, int multiplicity, int max_iter = 200,
                 double density_tol = 1.0e-8, double energy_tol = 1.0e-10)
        : solver_(charge, multiplicity, max_iter, density_tol, energy_tol) {}

    /// @brief Component name used in plan dumps and timing reports.
    std::string name() const override { return "am1_evaluator"; }

    /// @brief Turn the force computation on or off (lvalue overload).
    ///
    /// Off skips the finite-difference gradient entirely — the cost of an AM1
    /// step is dominated by it — and leaves the frame's force columns
    /// untouched. On is the default. The phase's own force request is ANDed
    /// with this, so an energy-only phase still skips the gradient.
    /// @param enabled True to compute and publish forces [eV/Å].
    /// @return `*this`, for chaining onto a named evaluator.
    AM1Evaluator& with_gradient(bool enabled) & noexcept {
        compute_force_ = enabled;
        return *this;
    }
    /// @brief Same, on a temporary — the `AM1Evaluator{}.with_gradient(true)`
    ///        form that is handed straight to `add()` / `set_evaluator()`.
    AM1Evaluator&& with_gradient(bool enabled) && noexcept {
        compute_force_ = enabled;
        return std::move(*this);
    }

    /// @brief One-time setup: size the NDDO engine for the frame's atom set.
    void build(const cpu::BuildContext& ctx) override;
    /// @brief One solve at the current geometry; results accumulate onto the
    ///        frame (energy, charges, and forces when requested).
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Results of this evaluator's most recent solve.
    const AM1Result& last_result() const { return solver_.last_result(); }
    /// @brief The owned method, for reading its parameters.
    const AM1& method() const { return solver_; }
    /// @brief The owned method, for reconfiguring it before the next `build()`.
    AM1& method() { return solver_; }
    /// @brief Is the gradient enabled on this evaluator?
    bool gradient_enabled() const noexcept { return compute_force_; }

  private:
    AM1 solver_;
    bool compute_force_ = true;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
