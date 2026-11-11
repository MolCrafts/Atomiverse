/// @file cpu/linalg/pulay.h
/// @brief Domain-independent Pulay equations and real AO commutator residual.
/// Pulay, Chem. Phys. Lett. 73, 393 (1980).
#pragma once

#include "atomiverse/cpu/linalg/matrix.h"
#include <cmath>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>

namespace ATV::cpu::linalg {

/// @brief Transform F P S - S P F into an orthonormal basis.
/// @param F Operator in the AO basis.
/// @param P Density matrix.
/// @param S AO overlap matrix.
/// @param X Orthogonalizer, satisfying X^T S X = I.
/// @return X^T (F P S - S P F) X, in the operator's units.
inline RealMatrix diis_error(const RealMatrix& F, const RealMatrix& P,
                             const RealMatrix& S, const RealMatrix& X) {
    const RealMatrix commutator = xt::linalg::dot(F, xt::linalg::dot(P, S))
                                  - xt::linalg::dot(S, xt::linalg::dot(P, F));
    return xt::linalg::dot(xt::transpose(X), xt::linalg::dot(commutator, X));
}

/// @brief Solve the unit-sum constrained Pulay equations from an error Gram matrix.
/// @param gram Symmetric m-by-m inner products of m residuals, m >= 1.
/// @return m dimensionless extrapolation coefficients whose sum is one.
/// @throws std::invalid_argument for an empty, nonsquare or nonfinite Gram matrix.
/// @throws std::runtime_error for a singular system or nonfinite coefficients.
inline RealVector pulay_coefficients(const RealMatrix& gram) {
    const std::size_t m = gram.shape(0);
    if (m == 0 || gram.shape(1) != m)
        throw std::invalid_argument(
            "Pulay: a nonempty square Gram matrix is required");
    RealMatrix bordered = xt::zeros<double>({m + 1, m + 1});
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            if (!std::isfinite(gram(i, j)))
                throw std::invalid_argument(
                    "Pulay: residual inner products must be finite");
            bordered(i, j) = gram(i, j);
        }
        bordered(i, m) = bordered(m, i) = -1.0;
    }
    RealMatrix rhs = xt::zeros<double>({m + 1, std::size_t{1}});
    rhs(m, 0) = -1.0;
    const RealMatrix solution = xt::linalg::solve(bordered, rhs);
    RealVector coefficients = xt::zeros<double>({m});
    for (std::size_t i = 0; i < m; ++i) {
        if (!std::isfinite(solution(i, 0)))
            throw std::runtime_error(
                "Pulay: nonfinite extrapolation coefficient");
        coefficients(i) = solution(i, 0);
    }
    return coefficients;
}

} // namespace ATV::cpu::linalg
