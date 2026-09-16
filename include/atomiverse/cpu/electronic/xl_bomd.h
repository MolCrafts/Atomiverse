/// @file cpu/electronic/xl_bomd.h
/// @brief XlBomd / XlBomdCplx — dissipative extended-Lagrangian
///        Born–Oppenheimer molecular dynamics (XL-BOMD) auxiliary-density
///        propagators (Niklasson's scheme).
///
/// **The problem.** In Born–Oppenheimer molecular dynamics (BOMD) the electrons
/// are re-converged at every nuclear step by a self-consistent-field (SCF)
/// iteration, which needs a starting guess for the electron density. Reusing the
/// previous step's converged density is the obvious guess, but it is *not* a
/// time-reversible operation — the guess always lags the nuclei — and the
/// resulting one-sided error pumps energy into the trajectory, so the total
/// energy drifts steadily over a long run. XL-BOMD removes the drift by giving
/// the guess its own reversible dynamics.
///
/// **The scheme.** An auxiliary density matrix \(P\) is propagated alongside the
/// nuclei and handed to the SCF as the initial guess each molecular-dynamics
/// (MD) step. With \(n\) the MD step index, \(P_n\) the auxiliary density at
/// step \(n\), \(D_n\) the SCF-converged density at step \(n\), and \(K\) the
/// dissipation order (history depth \(K+1\)):
///
///   P_{n+1} = 2 P_n − P_{n−1} + κ (D_n − P_n) + α Σ_{k=0}^{K} c_k P_{n−k}
///
/// The first three terms are the Verlet discretization of a harmonic equation
/// of motion \(\ddot P = \omega^2 (D[P] - P)\) — the auxiliary density behaves
/// like a mass on a spring whose equilibrium position is the true SCF density,
/// with \(\kappa = \omega^2 \Delta t^2\) the (dimensionless) Verlet coupling.
/// That part is exactly time-reversible. The last term is a weak dissipation
/// (strength \(\alpha \ll 1\), integer weights \(c_k\), both dimensionless) that
/// damps the numerical noise the SCF injects; it is the *only* time-irreversible
/// piece, of size \(O(\alpha)\). \((\kappa, \alpha, c_k)\) are the tabulated
/// rows of Table I in Niklasson et al., J. Chem. Phys. 130, 214109 (2009);
/// the underlying reversible scheme is Niklasson,
/// Phys. Rev. Lett. 100, 123004 (2008).
///
/// Contrast with the Kolafa always-stable predictor-corrector
/// (`cpu::electronic::ASPCPredictor`): that one *extrapolates* a polynomial
/// through past densities, whereas here the auxiliary density is a dynamical
/// variable with its own equation of motion.
///
/// **Units.** All matrices are one-particle density matrices in the
/// atomic-orbital (AO) basis — dimensionless — and the coefficients are
/// dimensionless, so this file carries no physical units: it is dense linear
/// algebra usable by any mean-field method that can hand over a converged
/// density. Consumed today by `cpu::gpw::GPWEvaluator` (`xl_bomd = true`), for
/// each spin/k-sampling combination it supports (Γ = the single \(k = 0\)
/// sampling point, where the density is real; RKS = restricted Kohn–Sham, one
/// orbital set for both spins; UKS = unrestricted Kohn–Sham, independent
/// \(\alpha\) and \(\beta\) spin channels):
/// - **Γ RKS:** total real \(P\) via `XlBomd`
/// - **Γ UKS:** one `XlBomd` per spin channel
/// - **multi-k RKS:** one `XlBomdCplx` per sampled \(k\) (complex \(P_k\))
/// - **multi-k UKS:** dual `XlBomdCplx` per \(k\) (\(P_k^\alpha, P_k^\beta\))
#pragma once

#include <complex>
#include <deque>
#include <vector>

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace electronic {

/// @brief Dissipative XL-BOMD auxiliary-density propagator (real AO density).
///
/// Lifecycle: construct with the dissipation order K, then once per MD step
/// (after the SCF solve) call `advance(D_scf)`. The first `advance` seeds
/// the history (stationary start: P ≡ D); every later call propagates. The
/// SCF guess for the *next* step is `aux()`.
class XlBomd {
  public:
    /// @brief Construct with the Niklasson dissipation order.
    /// @param k Dissipation order K (history depth K+1); Table I of
    ///          JCP 130, 214109 (2009) covers K ∈ [3, 9].
    /// @throws std::invalid_argument when k is outside [3, 9].
    explicit XlBomd(int k = 5);

    /// @brief Dissipation order K. @return K in [3, 9].
    int order() const { return k_; }
    /// @brief Verlet coupling κ (dimensionless). @return κ of Table I.
    double kappa() const { return kappa_; }
    /// @brief Dissipation strength α (dimensionless). @return α of Table I.
    double alpha() const { return alpha_; }
    /// @brief Dissipation weights c_0..c_K (dimensionless).
    /// @return K+1 coefficients of Table I (integer-valued).
    const std::vector<double>& c() const { return c_; }

    /// @brief Whether an auxiliary density is available (history seeded).
    /// @return true after the first `advance`.
    bool ready() const { return ready_; }

    /// @brief Current auxiliary density — the SCF initial guess for the
    ///        next MD step (same scale/convention as the SCF density fed to
    ///        `advance`: RKS total \(P\), or one UKS spin channel).
    /// @return P_{n+1} of the recursion above.
    /// @throws std::logic_error before the first `advance`.
    const linalg::RealMatrix& aux() const;

    /// @brief Record this step's SCF density and propagate the auxiliary
    ///        density one step.
    ///
    /// First call: seeds the whole history with `d_scf` (stationary start —
    /// the recursion's fixed point for a constant density). Later calls:
    /// applies the dissipative Verlet recursion.
    /// @param d_scf SCF-converged density matrix \(D_n\) (n_ao × n_ao,
    ///        dimensionless; n_ao = number of atomic-orbital basis functions).
    /// @throws std::invalid_argument when `d_scf` is empty or its shape
    ///         differs from the seeded history.
    void advance(const linalg::RealMatrix& d_scf);

    /// @brief Drop all history (geometry discontinuity / restart policy):
    ///        `ready()` becomes false and the next `advance` re-seeds.
    void reset();

  private:
    int k_;                 ///< dissipation order K.
    double kappa_ = 0.0;    ///< Verlet coupling κ.
    double alpha_ = 0.0;    ///< dissipation strength α.
    std::vector<double> c_; ///< dissipation weights c_0..c_K.
    /// Auxiliary-density history, newest first: P_n, P_{n−1}, …, P_{n−K}.
    std::deque<linalg::RealMatrix> hist_;
    linalg::RealMatrix aux_; ///< P_{n+1} (next-step SCF guess).
    bool ready_ = false;     ///< history seeded.
};

/// @brief Dissipative XL-BOMD propagator on complex Bloch densities \(P_k\).
///
/// Same Table-I recursion as `XlBomd`; real \((\kappa,\alpha,c_k)\) act
/// element-wise on `linalg::CplxMatrix`. A Hermitian history remains
/// Hermitian (up to floating-point roundoff). Used for multi-k RKS XL-BOMD
/// (one instance per sampled k) and multi-k UKS XL-BOMD (two instances per
/// k — α and β spin channels).
class XlBomdCplx {
  public:
    /// @brief Construct with the Niklasson dissipation order.
    /// @param k Dissipation order K ∈ [3, 9] (Table I).
    /// @throws std::invalid_argument when k is outside [3, 9].
    explicit XlBomdCplx(int k = 5);

    /// @brief Dissipation order K (history depth K+1).
    /// @return K in [3, 9] (dimensionless).
    int order() const { return k_; }
    /// @brief Verlet coupling κ (dimensionless), \(\kappa=\omega^2\Delta t^2\).
    /// @return κ of Table I for this K.
    double kappa() const { return kappa_; }
    /// @brief Dissipation strength α (dimensionless).
    /// @return α of Table I for this K.
    double alpha() const { return alpha_; }
    /// @brief Dissipation weights c_0..c_K (dimensionless).
    /// @return K+1 coefficients of Table I (integer-valued).
    const std::vector<double>& c() const { return c_; }
    /// @brief Whether an auxiliary density is available (history seeded).
    /// @return true after the first `advance`.
    bool ready() const { return ready_; }

    /// @brief Current auxiliary complex density \(P_{n+1}\) — the SCF initial
    ///        guess for the next MD step, in the same convention as the density
    ///        fed to `advance` (one k-point, one spin channel).
    /// @return \(P_{n+1}\) (n_ao × n_ao, dimensionless).
    /// @throws std::logic_error before the first `advance`.
    const linalg::CplxMatrix& aux() const;

    /// @brief Record this step's SCF density \(D_n\) at this k-point and
    ///        propagate the auxiliary density one step (first call seeds the
    ///        whole history with `d_scf`, as in `XlBomd::advance`).
    /// @param d_scf Complex Hermitian density (n_ao × n_ao, dimensionless).
    /// @throws std::invalid_argument on empty or shape-mismatched input.
    void advance(const linalg::CplxMatrix& d_scf);

    /// @brief Drop all history: `ready()` becomes false and the next `advance`
    ///        re-seeds.
    void reset();

  private:
    int k_;
    double kappa_ = 0.0;
    double alpha_ = 0.0;
    std::vector<double> c_;
    std::deque<linalg::CplxMatrix> hist_;
    linalg::CplxMatrix aux_;
    bool ready_ = false;
};

} // namespace electronic
} // namespace cpu
} // namespace ATV
