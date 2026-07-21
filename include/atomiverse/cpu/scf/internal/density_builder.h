/// @file cpu/scf/internal/density_builder.h
/// @brief internal::DensityBuilder — P = factor · C_occ · C_occᵀ.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component`.
///
/// Stateless thin wrapper around `build_density_matrix`, plus the spinor
/// (GHF) block utilities: `step` at dimension 2n builds the 2n×2n spinor
/// density from spinor MO coefficients, `spin_block` extracts one n×n spin
/// block from a 2n×2n spinor matrix, and `spinor_embed` lifts an n×n AO
/// matrix to its block-diagonal spinor form M⊗1₂ (blocked layout
/// [α-rows; β-rows]).
#pragma once

#include <stdexcept>
#include <string>

#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/linalg/density.h" // build_density_matrix
#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

class DensityBuilder {
  public:
    DensityBuilder() = default;

    /// @brief Pre-allocate workspaces sized for `n_basis`. No-op today.
    void prepare(int /*n_basis*/) {}

    /// @brief Build the density matrix from MO coefficients.
    /// @param C        MO coefficient matrix [n × n].
    /// @param n_occ    Number of occupied orbitals.
    /// @param n_basis  Basis dimension.
    /// @param factor   Occupancy factor (2 for closed-shell RHF, 1 per spin
    ///                 for unrestricted).
    /// @return Density matrix [n × n].
    RealMatrix step(const RealMatrix& C, int n_occ, int n_basis,
                    double factor = 1.0) const {
        return linalg::build_density_matrix(C, n_occ, n_basis, factor);
    }

    /// @brief Extract one n×n spin block from a 2n×2n spinor matrix in the
    ///        blocked layout [α-rows; β-rows] (GHF path).
    /// @param M_spinor Spinor matrix [2n × 2n] (units of M).
    /// @param bra Row spin component: 0 = α, 1 = β.
    /// @param ket Column spin component: 0 = α, 1 = β.
    /// @return M^{σσ'} [n × n], in the units of M.
    /// @throws std::invalid_argument when M is not square with even
    ///         dimension.
    static RealMatrix spin_block(const RealMatrix& M_spinor, int bra,
                                 int ket) {
        const std::size_t two_n = M_spinor.shape()[0];
        if (M_spinor.shape()[1] != two_n || two_n % 2 != 0)
            throw std::invalid_argument(
                "DensityBuilder::spin_block: expected a square 2n x 2n "
                "spinor matrix, got "
                + std::to_string(two_n) + "x"
                + std::to_string(M_spinor.shape()[1]));
        const std::size_t n = two_n / 2;
        const std::size_t r = (bra != 0) ? n : 0;
        const std::size_t c = (ket != 0) ? n : 0;
        return xt::view(M_spinor, xt::range(r, r + n), xt::range(c, c + n));
    }

    /// @brief Lift an n×n AO matrix to its 2n×2n block-diagonal spinor form
    ///        M⊗1₂ (blocked layout: identical α and β diagonal blocks, zero
    ///        spin-off-diagonal blocks). Used for S, Hcore and J, which are
    ///        spin-independent in real GHF.
    /// @param M AO matrix [n × n] (units of M).
    /// @return M⊗1₂ [2n × 2n], in the units of M.
    static RealMatrix spinor_embed(const RealMatrix& M) {
        const std::size_t n = M.shape()[0];
        RealMatrix out = xt::zeros<double>({2 * n, 2 * n});
        xt::view(out, xt::range(0, n), xt::range(0, n)) = M;
        xt::view(out, xt::range(n, 2 * n), xt::range(n, 2 * n)) = M;
        return out;
    }
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
