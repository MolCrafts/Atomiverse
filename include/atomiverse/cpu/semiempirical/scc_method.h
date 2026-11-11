/// @file cpu/semiempirical/scc_method.h
/// @brief One SCC iteration loop for DFTB and xTB.
#pragma once

#include <functional>

#include "atomiverse/cpu/driver/driver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Result of one `SccDriver::solve`.
///
/// `iterations` counts iterations that ran. `residual` is the charge residual
/// of the last one [e], or 0 when `max_iter` is 0. `converged` is the loop's
/// verdict: the residual Criterion returned `Converged`.
struct SccOutcome {
    int iterations = 0;
    double residual = 0;
    bool converged = false;
};

/// @brief `cpu::Driver<scc::Phase>` for one SCC charge loop.
///
/// Stack-allocate a fresh driver per solve. The five steps are the caller's
/// physics; this class owns only the iteration, the accept-versus-mix branch,
/// and the convergence verdict.
///
/// The Criterion is polled after the iteration, so a converged iteration must
/// not be mixed. The mix stage accepts when `residual < tol` and mixes
/// otherwise. The caller publishes the residual from its charges step with
/// `set_residual`. Each iteration starts from +\infty, so a charges step that
/// forgets to publish cannot converge.
///
/// `host` non-null runs the stages on that Runtime (`build_plan` / `step_plan`),
/// which is the parent of an in-progress `launch`. Null builds a scratch
/// Runtime for a solve that has no parent (a unit test, or an entry outside a
/// driver). The stages do not request forces or stress.
class SccDriver : public cpu::Driver<ATV::scc::Phase> {
  public:
    /// @brief Zero-argument body of one SCC stage.
    using Step = std::function<void()>;

    /// @brief Run at most `max_iter` SCC iterations.
    ///
    /// @param host Parent Runtime, or null for a standalone scratch Runtime.
    /// @param max_iter Iteration budget. The converging iteration counts.
    /// @param tol Charge residual tolerance [e]. Strict `<`.
    /// @param shift Build the potential and the shifted Hamiltonian.
    /// @param diagonalize Diagonalize and build the density.
    /// @param charges Mulliken reduction. Must call `set_residual`.
    /// @param accept Copy the output charges into the input (no-op when the
    ///        density step already stored them).
    /// @param mix Anderson-mix the output charges into the input.
    /// @return Iterations, last residual [e], and the loop verdict.
    SccOutcome solve(cpu::Runtime* host, int max_iter, double tol, Step shift,
                     Step diagonalize, Step charges, Step accept, Step mix);

    /// @brief Publish the charge residual of the current iteration [e].
    /// @param residual Max |q_out − q_in| [e].
    void set_residual(double residual) { residual_ = residual; }

    /// @brief Charge residual of the current iteration [e].
    /// @return The value last published, or +\infty before `set_residual`.
    double residual() const { return residual_; }

  protected:
    void validate_components() const override {}
    bool phase_wants_forces(ATV::scc::Phase) const override { return false; }
    bool phase_wants_stress(ATV::scc::Phase) const override { return false; }

  private:
    double residual_ = 0;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
