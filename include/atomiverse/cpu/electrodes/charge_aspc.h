/// @file cpu/electrodes/charge_aspc.h
/// @brief Kolafa ASPC predictor-corrector for electrode charge vectors.
///
/// MetalWalls warm-starts the iterative (CG) electrode charge solve from a
/// short history of prior \(Q\) vectors, using Kolafa's ASPC predictor
/// (J. Comput. Chem. 25, 335 (2004), Eq. 10) applied to flat charge vectors
/// rather than density matrices. MetalWalls c8ad175d hard-codes the \(K=5\)
/// row (six history vectors, `src/electrode_charge.F90:572-577`); here the
/// order is a constructor argument and \(K=5\) reproduces it exactly.
///
/// The predictor is **not** Lagrange polynomial extrapolation: ASPC is exact
/// only through degree 1, trading higher-order accuracy for the unconditional
/// stability and time-reversibility that the corrector blend relies on.
///
/// No public `*Config` bag; order is a constructor argument.
#pragma once

#include <deque>
#include <vector>

namespace ATV {
namespace cpu {
namespace electrodes {

/// @brief Always-Stable Predictor-Corrector (ASPC) for electrode charges [e].
///
/// Usage (with an iterative BO/CG charge solve):
/// @code
/// electrodes::ChargeASPC aspc(/*order=*/2);
/// for (each MD step) {
///     std::vector<double> Q_guess;
///     if (aspc.predict(Q_guess)) {
///         // warm-start CG from Q_guess
///     }
///     // … converge Q_scf …
///     aspc.record(Q_scf);
/// }
/// @endcode
class ChargeASPC {
  public:
    /// @brief Construct with Kolafa ASPC order \(K\).
    /// @param order Order \(K \ge 0\). Needs \(K+1\) history points to predict;
    ///              \(K=5\) is MetalWalls' production choice.
    explicit ChargeASPC(int order = 2);

    /// @brief Record a converged electrode charge vector into history.
    /// @param Q_scf Converged charges [e], length fixed after first record.
    /// @throws std::invalid_argument if length mismatches an established history.
    void record(const std::vector<double>& Q_scf);

    /// @brief Predict the next-step charge vector with the ASPC coefficients.
    /// @param[out] Q_predicted Extrapolated charges [e] (resized).
    /// @return True if \(K+1\) history points were available; false otherwise.
    bool predict(std::vector<double>& Q_predicted) const;

    /// @brief Kolafa corrector blend for time-reversibility.
    ///
    /// \(Q_{\mathrm{corr}} = \frac{K+2}{2K+3} Q_{\mathrm{scf}}
    ///   + \frac{K+1}{2K+3} Q_{\mathrm{pred}}\).
    ///
    /// @param Q_scf       Converged charges [e].
    /// @param Q_predicted Predictor output [e].
    /// @param[out] Q_corrected Blended vector to record [e].
    void correct(const std::vector<double>& Q_scf,
                 const std::vector<double>& Q_predicted,
                 std::vector<double>& Q_corrected) const;

    /// @brief Drop all history (geometry jump / restart).
    void reset();

    /// @brief Number of stored history vectors.
    /// @return History length.
    int history_size() const { return static_cast<int>(history_.size()); }

    /// @brief ASPC order \(K\).
    /// @return Order passed to the constructor.
    int order() const { return K_; }

    /// @brief ASPC coefficient \(\omega_k\) for history index \(k\)
    ///        (0 = oldest of the \(K+1\) window used for prediction), i.e.
    ///        Kolafa's \(B_{K+1-k}\).
    /// @param k Index in \([0, K]\).
    /// @return Coefficient \(\omega_k\).
    double coefficient(int k) const;

  private:
    int K_;
    std::deque<std::vector<double>> history_; ///< Newest last.
    std::vector<double> omega_;               ///< ASPC coeffs, length K+1.
    int dim_ = -1; ///< Established charge-vector length; -1 until first record.

    void compute_coefficients_();
};

} // namespace electrodes
} // namespace cpu
} // namespace ATV
