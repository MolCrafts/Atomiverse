/// @file cpu/scf/aspc_predictor.h
/// @brief Always-Stable Predictor-Corrector (ASPC) for density matrix extrapolation.
///
/// Implements the Kolafa predictor-corrector scheme for second-generation
/// Car-Parrinello AIMD. Extrapolates density matrices from previous MD steps
/// to provide a good initial guess, reducing the number of SCF iterations
/// needed per step.
///
/// Reference: Kolafa, J. Comput. Chem. 25, 335 (2004)
///            Kuhne et al., PRL 98, 066401 (2007)
#pragma once

#include <deque>
#include <vector>
#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief ASPC density matrix predictor-corrector.
///
/// Usage:
/// @code{.cpp}
/// ASPCPredictor pred(3);  // order K=3
/// for (int step = 0; step < n_steps; ++step) {
///     xt::xtensor<double, 2> P_guess;
///     if (pred.predict(P_guess)) {
///         // Use P_guess as SCF initial guess
///         auto result = rhf_solve_with_guess(integrals, P_guess, tol, max_iter);
///         pred.correct(result.P, P_guess, P_corrected);
///         pred.record(P_corrected);
///     } else {
///         // Not enough history, do full SCF
///         auto result = SCFSolver(integrals, tol, max_iter);
///         pred.record(result.P);
///     }
/// }
/// @endcode
class ASPCPredictor {
  public:
    /// @brief Construct with extrapolation order.
    /// @param order Lagrange extrapolation order K (default 3). Uses K+1 history points.
    explicit ASPCPredictor(int order = 3);

    /// @brief Record a converged density matrix into history.
    /// @param P_scf Converged density matrix from SCF.
    void record(const xt::xtensor<double, 2>& P_scf);

    /// @brief Predict density for the next step via Lagrange extrapolation.
    /// @param[out] P_predicted Extrapolated density matrix.
    /// @return True if prediction was made (sufficient history), false otherwise.
    bool predict(xt::xtensor<double, 2>& P_predicted) const;

    /// @brief Blend corrector result with predictor for time-reversibility.
    ///
    /// P_corrected = (K+2)/(2K+3) * P_scf + (K+1)/(2K+3) * P_predicted
    ///
    /// @param P_scf       SCF-converged density at current step.
    /// @param P_predicted Predictor output (from predict()).
    /// @param[out] P_corrected Blended density to record.
    void correct(const xt::xtensor<double, 2>& P_scf,
                 const xt::xtensor<double, 2>& P_predicted,
                 xt::xtensor<double, 2>& P_corrected) const;

    /// @brief Reset history (e.g., after geometry discontinuity or restart).
    void reset();

    /// @brief Number of history points currently stored.
    /// @return Count of stored density matrices.
    int history_size() const { return static_cast<int>(history_.size()); }

    /// @brief Extrapolation order K.
    /// @return Lagrange extrapolation order.
    int order() const { return K_; }

  private:
    int K_; ///< Extrapolation order.
    std::deque<xt::xtensor<double, 2>>
        history_;               ///< Stored density matrices (newest last).
    std::vector<double> omega_; ///< Lagrange extrapolation coefficients.

    /// @brief Compute Lagrange extrapolation coefficients for order K.
    void compute_coefficients();
};

} // namespace scf
} // namespace cpu
} // namespace ATV
