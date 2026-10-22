/// @file cpu/gpw/internal/diis.h
/// @brief internal::GpwDiis — single-channel Pulay DIIS for the restricted GPW KS matrix.
///
/// **Internal GPW helper.** Does NOT inherit `cpu::Component` and is not
/// registered into any outer Driver Plan. Owned by the GPW SCF loop (`GPW`)
/// and by tests (via direct include). Self-contained in `cpu::gpw::internal`;
/// it reuses only `cpu::linalg` matrix types (never `cpu::scf::internal`), per
/// the GPW domain-isolation rule.
///
/// Pulay's Direct Inversion of the Iterative Subspace (DIIS) accelerates the
/// closed-shell GPW self-consistency by extrapolating the Kohn-Sham (Fock)
/// matrix from a history of (H, error) pairs. The error vector is the
/// commutator residual in the Löwdin-orthonormal basis,
/// `e = X^T (H P S - S P H) X` (X = S^{-1/2}), which vanishes at
/// self-consistency. This replaces the slow linear density mixing that fails to
/// converge transition-metal / semicore systems (Zn-q12, Na-q9) and does not
/// scale to large cells.
///
/// References:
///   - Pulay, Chem. Phys. Lett. 73, 393 (1980)
///   - Pulay, J. Comp. Chem. 3, 556 (1982)
#pragma once

#include <deque>
#include <stdexcept>

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

using linalg::RealMatrix;

/// @brief Single-channel Pulay DIIS accelerator for the restricted GPW KS matrix.
///
/// Stores a depth-bounded history of (H, error) pairs, solves the Pulay
/// B-matrix system `B_ij = Tr(e_i e_j)` (augmented with the unit-sum Lagrange
/// constraint), and returns the extrapolated KS matrix `H_diis = Σ_i c_i H_i`.
/// Before two history vectors exist it returns the latest H unchanged (natural
/// warm-up). Falls back to the latest H if the B-matrix solve is singular.
class GpwDiis {
  public:
    /// @param max_vectors Maximum number of stored (H, error) pairs (>= 1).
    explicit GpwDiis(int max_vectors = 6) : max_vectors_(max_vectors) {
        if (max_vectors < 1)
            throw std::invalid_argument(
                "GpwDiis: history depth must be positive");
    }

    /// @brief Store a new (H, error) pair and return the extrapolated KS matrix.
    ///
    /// @param H Current KS (Fock) matrix [Hartree].
    /// @param e Current DIIS error matrix (from linalg::diis_error).
    /// @return Extrapolated KS matrix [Hartree] (latest H during warm-up / on
    ///         a singular subspace).
    RealMatrix step(const RealMatrix& H, const RealMatrix& e);

    /// @brief Drop all stored (H, error) pairs (e.g. after an energy ascent).
    void clear() {
        h_hist_.clear();
        e_hist_.clear();
    }

  private:
    int max_vectors_;
    std::deque<RealMatrix> h_hist_;
    std::deque<RealMatrix> e_hist_;
};

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
