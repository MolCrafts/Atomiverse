/// @file cpu/scf/internal/dual_convergence_check.h
/// @brief internal::DualConvergenceCheck — PySCF-aligned |ΔE| ∧ ‖err‖ check.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component`.
///
/// Reports whether an SCF iteration has converged using PySCF's standard
/// dual criterion:
///   - `|E_total - E_prev| < tol_energy`
///   - `‖err‖_F < tol_grad`     (Frobenius norm, combined α/β when applicable)
///
/// Iteration 0 always reports `converged=false` (no `E_prev` yet).
///
/// Stateful: caches `prev_energy` between `step()` calls. Reset at the
/// start of each SCF loop (`MeanFieldMethod::run` / `ScfConverged` setup).
#pragma once

#include <cmath>
#include <limits>

#include <xtensor/reducers/xnorm.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

struct ConvergenceVerdict {
    bool converged = false; ///< True iff dual criteria both satisfied.
    double dE = 0.0;        ///< E_total − E_prev (∞ on iter 0).
    double err_norm = 0.0;  ///< ‖err‖_F (combined α + β, Frobenius).
};

class DualConvergenceCheck {
  public:
    /// @param tol_energy  Energy tolerance (Hartree).
    /// @param tol_grad    DIIS-error gradient tolerance (Frobenius norm).
    DualConvergenceCheck(double tol_energy, double tol_grad)
        : tol_energy_(tol_energy), tol_grad_(tol_grad) {}

    /// @brief Reset internal state at the start of a new SCF loop.
    void reset() {
        prev_energy_ = 0.0;
        has_prev_ = false;
    }

    /// @brief Evaluate convergence for the current iteration.
    /// @param E_total    Total electronic + nuclear energy this iteration (Hartree).
    /// @param err_alpha  α-channel DIIS error matrix.
    /// @param err_beta   β-channel DIIS error matrix; pass `nullptr` (or the
    ///                   same as err_alpha) for restricted SCF.
    /// @return Verdict (converged flag + diagnostics).
    ConvergenceVerdict step(double E_total, const RealMatrix& err_alpha,
                            const RealMatrix* err_beta = nullptr) {
        ConvergenceVerdict v;
        v.dE = has_prev_ ? E_total - prev_energy_
                         : std::numeric_limits<double>::infinity();

        const double ea = xt::norm_l2(err_alpha)();
        if (err_beta != nullptr) {
            const double eb = xt::norm_l2(*err_beta)();
            v.err_norm = std::sqrt(ea * ea + eb * eb);
        } else {
            v.err_norm = ea;
        }

        v.converged = has_prev_ && (std::abs(v.dE) < tol_energy_)
                      && (v.err_norm < tol_grad_);

        prev_energy_ = E_total;
        has_prev_ = true;
        return v;
    }

  private:
    double tol_energy_;
    double tol_grad_;
    double prev_energy_ = 0.0;
    bool has_prev_ = false;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
