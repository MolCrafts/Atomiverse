/// @file cpu/scf/internal/diis_mixer.h
/// @brief internal::DIISMixer — combined DIIS accelerator for spin-resolved SCF.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component` and is
/// not registered into any outer Driver Plan. Owned by the DIIS / mix stage
/// inside the driver-ized RHF/UHF/RKS/UKS loop (and by tests, via direct
/// include of this header).
///
/// Pulay's Direct Inversion of the Iterative Subspace (DIIS) with combined
/// error metric across alpha/beta spin channels. Shared extrapolation
/// coefficients ensure consistent convergence for both spins.
///
/// References:
///   - Pulay, Chem. Phys. Lett. 73, 393 (1980)
///   - Pulay, J. Comp. Chem. 3, 556 (1982)
#pragma once

#include <deque>
#include <utility>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Compute DIIS error vector in the orthonormal basis.
///
/// Matches PySCF: e = X^T (F P S - S P F) X, where X is the canonical
/// orthogonalization matrix (X^T S X = I). At self-consistency, e -> 0.
inline RealMatrix diis_error(const RealMatrix& F, const RealMatrix& P,
                             const RealMatrix& S, const RealMatrix& X) {
    RealMatrix commutator = xt::linalg::dot(F, xt::linalg::dot(P, S))
                            - xt::linalg::dot(S, xt::linalg::dot(P, F));
    return xt::linalg::dot(xt::transpose(X), xt::linalg::dot(commutator, X));
}

/// @brief Combined DIIS accelerator for spin-resolved Fock matrices.
///
/// Stores history of Fock matrices and DIIS error vectors for both spin
/// channels. Solves the Pulay B-matrix equations with combined error
/// metric: `B_ij = Tr(e^α_i · e^α_j) + Tr(e^β_i · e^β_j)`.
///
/// For restricted (single-spin), pass the same F/e for both channels or
/// use the single-channel overload. The combined B matrix naturally
/// reduces to standard single-channel DIIS when e_alpha == e_beta.
class DIISMixer {
  public:
    /// @param max_vectors Maximum number of stored (F, e) pairs.
    explicit DIISMixer(int max_vectors = 6) : max_vectors_(max_vectors) {}

    /// @brief Store a new Fock/error pair and return extrapolated Fock matrices.
    std::pair<RealMatrix, RealMatrix> step(const RealMatrix& F_alpha,
                                           const RealMatrix& e_alpha,
                                           const RealMatrix& F_beta,
                                           const RealMatrix& e_beta);

    /// @brief Single-channel convenience (for restricted SCF).
    RealMatrix step(const RealMatrix& F, const RealMatrix& e);

    /// @brief Number of stored Fock/error pairs.
    int size() const { return static_cast<int>(Fa_hist_.size()); }

    /// @brief Clear all history.
    void reset() {
        Fa_hist_.clear();
        Fb_hist_.clear();
        ea_hist_.clear();
        eb_hist_.clear();
    }

  private:
    int max_vectors_;
    std::deque<RealMatrix> Fa_hist_, Fb_hist_;
    std::deque<RealMatrix> ea_hist_, eb_hist_;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
