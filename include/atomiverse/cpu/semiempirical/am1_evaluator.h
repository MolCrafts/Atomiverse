/// @file cpu/semiempirical/am1_evaluator.h
/// @brief AM1 Evaluator component for Atomiverse drivers.
///
/// Mirrors the existing SCF/PW/GPW shape: `AM1Evaluator` is the Component that
/// plugs into outer drivers, while `AM1Solver` is the inner solver it owns.
#pragma once

#include <string>
#include <utility>

#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief AM1 semiempirical Evaluator component (wraps an `AM1Solver`).
///
/// Kind: `cpu::Semiempirical`. Inner solver: `AM1Solver` → `NDDOSolver`.
/// Bind explicitly with `.on(phase).add(...)` (no default_bindings).
class AM1Evaluator : public cpu::Semiempirical {
  public:
    /// @brief Construct with an AM1 configuration.
    /// @param config Solver configuration (charge, SCF tolerance, ...).
    explicit AM1Evaluator(AM1Config config = {}) : solver_(std::move(config)) {}

    /// @brief Component name.
    /// @return "am1_evaluator".
    std::string name() const override { return "am1_evaluator"; }

    /// @brief Toggle nuclear force writes (default on, like `cpu::scf::SCF`).
    ///
    /// When enabled and `ctx.forces_requested`, `launch()` runs the NDDO
    /// analytic gradient and writes `kAtom_fx/fy/fz` [eV/Å].
    /// @param enabled Whether to compute forces after SCF.
    /// @return `*this` for chaining.
    AM1Evaluator& with_gradient(bool enabled) & noexcept {
        compute_force_ = enabled;
        return *this;
    }
    /// @copydoc with_gradient(bool) &
    AM1Evaluator&& with_gradient(bool enabled) && noexcept {
        compute_force_ = enabled;
        return std::move(*this);
    }

    /// @brief Build the owned solver against the reference frame (one-time).
    /// @param ctx Build context (runtime services + frame).
    void build(const cpu::BuildContext& ctx) override;
    /// @brief Run AM1 SCF; write energy [eV], charges [e], optional forces
    ///        [eV/Å].
    /// @param ctx Launch context (runtime services + frame + stage/step).
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Converged result of the most recent launch.
    /// @return The solver's last `AM1Result` (energies, charges, forces).
    const AM1Result& last_result() const { return solver_.last_result(); }
    /// @brief Solver configuration in effect.
    /// @return The `AM1Config` this evaluator was constructed with.
    const AM1Config& config() const { return solver_.config(); }
    /// @brief Whether force computation is enabled.
    bool gradient_enabled() const noexcept { return compute_force_; }

  private:
    AM1Solver solver_;
    bool compute_force_ = true;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
