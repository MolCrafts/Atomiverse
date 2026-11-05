/// @file src/cpu/semiempirical/nddo_density.cpp
/// @brief NDDO reference densit.F90 port for NDDO density matrices.

#include "atomiverse/cpu/semiempirical/nddo_density.h"

#include <algorithm>
#include <stdexcept>

#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

extern "C" {
// Fortran BLAS rank-k update, the vocabulary of the zgemm_ binding in
// cpu/linalg/orbital_tangent.cpp; xtensor-blas does not declare dsyrk_.
void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda,
            const double* beta, double* c, const int* ldc);
}

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

/// Accumulate alpha · Σ_{k in [first, last)} C(i,k) C(j,k) into the upper
/// triangle of the column-major dense matrix (i <= j). Row-major C with row
/// stride mdim is, read column-major, Cᵀ with leading dimension mdim, so the
/// sum is (Cᵀ)ᵀ(Cᵀ) restricted to the rows [first, last) of Cᵀ.
void accumulate_range(const std::vector<double>& c, int mdim, int norbs,
                      int first0, int last0, double alpha, double beta,
                      std::vector<double>& dense) {
    const int k = last0 - first0;
    if (k <= 0) {
        if (beta == 0.0)
            std::fill(dense.begin(), dense.end(), 0.0);
        return;
    }
    // Direct Fortran BLAS: the cxxblas template overload resolves to its
    // generic scalar loop here, which costs more than the whole BLAS call.
    const char upper = 'U', transpose = 'T';
    dsyrk_(&upper, &transpose, &norbs, &k, &alpha, c.data() + first0, &mdim,
           &beta, dense.data(), &norbs);
}

void validate_density_input(const std::vector<double>& coefficients,
                            const NDDODensityInput& input) {
    if (input.mdim <= 0 || input.norbs <= 0) {
        throw std::runtime_error("NDDO density dimensions must be positive");
    }
    if (input.mdim < input.norbs) {
        throw std::runtime_error("NDDO density mdim must be >= norbs");
    }
    if (input.n_occupied < 0 || input.n_occupied > input.norbs
        || input.n_fractional < input.n_occupied
        || input.n_fractional > input.norbs) {
        throw std::runtime_error("NDDO density occupation bounds are invalid");
    }
    const std::size_t expected = static_cast<std::size_t>(input.mdim)
                                 * static_cast<std::size_t>(input.mdim);
    if (coefficients.size() < expected) {
        throw std::runtime_error(
            "NDDO density coefficient matrix is smaller than mdim*mdim");
    }
}

} // namespace

void NDDODensityBuilder::build_packed(const std::vector<double>& coefficients,
                                      const NDDODensityInput& input,
                                      std::vector<double>& p) {
    validate_density_input(coefficients, input);
    if (&coefficients == &p)
        throw std::invalid_argument("NDDO density output aliases coefficients");

    const int norbs2 = input.norbs / 2;
    double sign = 1.0;
    double frac = input.fractional_occupancy;
    double diagonal_constant = 0.0;
    int nl2 = 1;
    int nu2 = input.n_occupied;
    int nl1 = input.n_occupied + 1;
    int nu1 = input.n_fractional;

    if (input.n_occupied != 0 && input.n_fractional > norbs2
        && input.mode != 2) {
        sign = -1.0;
        frac = input.full_occupancy - input.fractional_occupancy;
        diagonal_constant = input.full_occupancy;
        nl2 = input.n_fractional + 1;
        nu2 = input.norbs;
        nl1 = input.n_occupied + 1;
        nu1 = input.n_fractional;
    }

    const int n = input.norbs;
    dense_.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    // densit.F90: P = sign · (full · Σ_{nl2..nu2} + frac · Σ_{nl1..nu1}) + d.
    accumulate_range(coefficients, input.mdim, n, nl2 - 1, nu2,
                     sign * input.full_occupancy, 0.0, dense_);
    accumulate_range(coefficients, input.mdim, n, nl1 - 1, nu1, sign * frac,
                     1.0, dense_);
    p.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1)
             / 2u);
    std::size_t idx = 0;
    for (int i = 0; i < n; ++i) {
        // Upper triangle column i holds P(j, i) for j <= i, contiguous in j.
        const double* column = dense_.data() + static_cast<std::size_t>(i) * n;
        for (int j = 0; j <= i; ++j, ++idx)
            p[idx] = column[j];
        p[idx - 1] += diagonal_constant;
    }
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
