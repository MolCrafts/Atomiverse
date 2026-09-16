/// @file cpu/electronic/aspc_predictor.h
/// @brief Always-Stable Predictor-Corrector (ASPC) for density matrix
///        extrapolation across molecular-dynamics steps.
///
/// **The problem.** Each step of ab initio molecular dynamics (AIMD — molecular
/// dynamics in which the forces come from an electronic-structure calculation
/// rather than a fitted force field) needs a starting guess for the
/// self-consistent-field (SCF) iteration, the loop that makes the electronic
/// Hamiltonian consistent with the density it produces. A better guess means
/// fewer SCF iterations, and the nuclei move smoothly, so the density matrices
/// of the previous steps predict the next one well.
///
/// **The scheme.** Kolafa's always-stable predictor-corrector fits a Lagrange
/// interpolating polynomial through the \(K+1\) most recent density matrices and
/// evaluates it one step beyond the newest — a pure extrapolation. Label the
/// retained densities \(P^{(0)}\) (oldest kept) … \(P^{(K)}\) (newest); the
/// predictor is
///
///   P_pred = Σ_{j=0}^{K} ω_j P^{(j)},
///   ω_j = Π_{i=0..K, i≠j} ((K+1) − i) / (j − i)
///
/// with \(K\) the extrapolation order and \(\omega_j\) the (dimensionless)
/// Lagrange weights for evaluating at abscissa \(K+1\). Extrapolation alone is
/// not time-reversible, so a *corrector* step blends the SCF result obtained
/// from that guess back toward the prediction,
///
///   P_corr = (K+2)/(2K+3) · P_scf + (K+1)/(2K+3) · P_pred,
///
/// and it is `P_corr` — not the raw SCF density — that is recorded into the
/// history. That blend is what makes the scheme "always stable": the recorded
/// trajectory of density matrices stays reversible enough that the energy does
/// not drift. This is the density predictor of second-generation
/// Car–Parrinello molecular dynamics (Kühne et al.), where the corrector is a
/// deliberately *partial* SCF solve of only a few iterations.
///
/// **Paths and units.** All matrices are one-particle density matrices in the
/// atomic-orbital (AO) basis, hence dimensionless, and the weights are
/// dimensionless too — this file carries no physical units. `ASPCPredictor`
/// serves the real \(\Gamma\) path (Γ = the single \(k = 0\) sampling point of a
/// periodic cell, where the density can be taken real). `ASPCPredictorCplx`
/// serves the multi-k path, acting element-wise with the *same* real weights on
/// each Hermitian Bloch density \(P_k\) (one instance per sampled k-point).
///
/// Method-agnostic dense algebra: consumed today by
/// `cpu::gpw::GPWEvaluator` (`second_gen_cp = true`) and by the Gaussian-basis
/// `cpu::scf::MeanFieldMethod` density predictor.
///
/// Reference: Kolafa, J. Comput. Chem. 25, 335 (2004), Eqs. 7-11
///            Kühne, Krack, Mohamed & Parrinello,
///            Phys. Rev. Lett. 98, 066401 (2007)
#pragma once

#include <complex>
#include <deque>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace electronic {

/// @brief ASPC density matrix predictor-corrector (real AO density).
///
/// Usage (the caller owns the SCF solve; this type only extrapolates):
/// @code{.cpp}
/// ASPCPredictor pred(3);  // order K=3
/// for (int step = 0; step < n_steps; ++step) {
///     xt::xtensor<double, 2> P_guess;
///     if (pred.predict(P_guess)) {
///         // Warm step: solve from P_guess with a small iteration budget,
///         // then blend the converged density back for time-reversibility.
///         const auto P_scf = /* corrector solve seeded with P_guess */;
///         xt::xtensor<double, 2> P_corrected;
///         pred.correct(P_scf, P_guess, P_corrected);
///         pred.record(P_corrected);
///     } else {
///         // Cold step: not enough history — full SCF, record its density.
///         pred.record(/* converged density */);
///     }
/// }
/// @endcode
class ASPCPredictor {
  public:
    /// @brief Construct with extrapolation order.
    /// @param order Lagrange extrapolation order K (dimensionless, default 3).
    ///        Prediction needs K+1 history points; the deque keeps at most K+2.
    /// @throws std::invalid_argument when order < 0.
    explicit ASPCPredictor(int order = 3);

    /// @brief Record a density matrix into history (newest last), dropping the
    ///        oldest once more than K+2 are stored.
    ///
    /// Feed the **corrector output** on warm steps and the converged SCF density
    /// on cold steps — recording the raw SCF density on a warm step throws away
    /// the reversibility the corrector buys.
    /// @param P_scf Density matrix to store (n_ao × n_ao, dimensionless;
    ///        n_ao = number of atomic-orbital basis functions).
    void record(const xt::xtensor<double, 2>& P_scf);

    /// @brief Predict the next step's density by Lagrange extrapolation over the
    ///        K+1 newest history entries (formula in the file header).
    /// @param[out] P_predicted Extrapolated density matrix (n_ao × n_ao,
    ///        dimensionless); untouched when the history is too short.
    /// @return true when `history_size() >= K+1` and a prediction was written;
    ///         false otherwise (the caller must then do a cold SCF solve).
    bool predict(xt::xtensor<double, 2>& P_predicted) const;

    /// @brief Blend corrector result with predictor for time-reversibility.
    ///
    /// P_corrected = (K+2)/(2K+3) * P_scf + (K+1)/(2K+3) * P_predicted
    ///
    /// The two weights (dimensionless) sum to 1, so the blend preserves the
    /// electron count of the inputs.
    /// @param P_scf       SCF-converged density at the current step
    ///                    (n_ao × n_ao, dimensionless).
    /// @param P_predicted Predictor output (from `predict()`), same shape.
    /// @param[out] P_corrected Blended density to hand to `record()`.
    void correct(const xt::xtensor<double, 2>& P_scf,
                 const xt::xtensor<double, 2>& P_predicted,
                 xt::xtensor<double, 2>& P_corrected) const;

    /// @brief Reset history (e.g., after geometry discontinuity or restart), so
    ///        the next `predict()` returns false until K+1 records accumulate.
    void reset();

    /// @brief Number of history points currently stored.
    /// @return Count of stored density matrices, 0 … K+2.
    int history_size() const { return static_cast<int>(history_.size()); }

    /// @brief Extrapolation order K.
    /// @return Lagrange extrapolation order (dimensionless, ≥ 0).
    int order() const { return K_; }

  private:
    int K_; ///< Extrapolation order.
    std::deque<xt::xtensor<double, 2>>
        history_;               ///< Stored density matrices (newest last).
    std::vector<double> omega_; ///< Lagrange extrapolation coefficients.

    /// @brief Compute Lagrange extrapolation coefficients for order K.
    void compute_coefficients();
};

/// @brief ASPC on complex Hermitian Bloch densities \(P_k\) — the multi-k
///        second-generation Car–Parrinello (SGCP) path.
///
/// Same Kolafa Lagrange weights \(\omega_j\) and corrector weights as
/// `ASPCPredictor` (real, dimensionless), applied element-wise to
/// `linalg::CplxMatrix`. Because the weights are real, a Hermitian history stays
/// Hermitian up to floating-point roundoff. Used by `GPWEvaluator` with
/// `second_gen_cp` and a k-mesh beyond Γ: one instance per sampled k-point, two
/// per k-point for unrestricted Kohn–Sham (α and β spin channels).
class ASPCPredictorCplx {
  public:
    /// @brief Construct with extrapolation order.
    /// @param order Lagrange order K (dimensionless, ≥ 0); prediction needs K+1
    ///        history points, the deque keeps at most K+2.
    /// @throws std::invalid_argument when order < 0.
    explicit ASPCPredictorCplx(int order = 3);

    /// @brief Record a (post-corrector, or cold-SCF on the first steps) density
    ///        into history, dropping the oldest beyond K+2 entries.
    /// @param P_scf Complex Hermitian density at one k-point (n_ao × n_ao,
    ///        dimensionless).
    void record(const linalg::CplxMatrix& P_scf);

    /// @brief Predict the next-step density by Lagrange extrapolation over the
    ///        K+1 newest entries (formula in the file header).
    /// @param[out] P_predicted Extrapolated complex density (n_ao × n_ao,
    ///        dimensionless); untouched when the history is too short.
    /// @return true when `history_size() >= K+1`; false otherwise.
    bool predict(linalg::CplxMatrix& P_predicted) const;

    /// @brief Kolafa blend \(P_\mathrm{corr} = w_\mathrm{scf} P_\mathrm{scf}
    ///        + w_\mathrm{pred} P_\mathrm{pred}\) with
    ///        \(w_\mathrm{scf} = (K+2)/(2K+3)\),
    ///        \(w_\mathrm{pred} = (K+1)/(2K+3)\) (dimensionless, summing to 1).
    /// @param P_scf       SCF density at the current step (n_ao × n_ao,
    ///                    dimensionless).
    /// @param P_predicted Predictor output from `predict()`, same shape.
    /// @param[out] P_corrected Blended density to `record()`.
    void correct(const linalg::CplxMatrix& P_scf,
                 const linalg::CplxMatrix& P_predicted,
                 linalg::CplxMatrix& P_corrected) const;

    /// @brief Drop all history (geometry discontinuity / restart), so the next
    ///        `predict()` returns false until K+1 records accumulate.
    void reset();

    /// @brief Number of stored densities.
    /// @return History length, 0 … K+2.
    int history_size() const { return static_cast<int>(history_.size()); }

    /// @brief Extrapolation order K.
    /// @return Lagrange extrapolation order (dimensionless, ≥ 0).
    int order() const { return K_; }

  private:
    int K_;
    std::deque<linalg::CplxMatrix> history_;
    std::vector<double> omega_;

    void compute_coefficients();
};

} // namespace electronic
} // namespace cpu
} // namespace ATV
