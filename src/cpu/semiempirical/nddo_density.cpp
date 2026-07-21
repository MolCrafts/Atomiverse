/// @file src/cpu/semiempirical/nddo_density.cpp
/// @brief NDDO reference densit.F90 port for NDDO density matrices.

#include "atomiverse/cpu/semiempirical/nddo_density.h"

#include <stdexcept>

#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

double coeff(const std::vector<double>& c, int mdim, int row1, int col1) {
    return c[static_cast<std::size_t>(row1 - 1) * static_cast<std::size_t>(mdim)
             + static_cast<std::size_t>(col1 - 1)];
}

double column_product_sum(const std::vector<double>& c, int mdim, int i1,
                          int j1, int first_col1, int last_col1) {
    if (first_col1 > last_col1) {
        return 0.0;
    }
    double sum = 0.0;
    for (int k1 = first_col1; k1 <= last_col1; ++k1) {
        sum += coeff(c, mdim, i1, k1) * coeff(c, mdim, j1, k1);
    }
    return sum;
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

std::vector<double>
NDDODensityBuilder::build_packed(const std::vector<double>& coefficients,
                                 const NDDODensityInput& input) const {
    validate_density_input(coefficients, input);

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

    std::vector<double> p(static_cast<std::size_t>(input.norbs)
                              * static_cast<std::size_t>(input.norbs + 1) / 2u,
                          0.0);
    for (int i1 = 1; i1 <= input.norbs; ++i1) {
        for (int j1 = 1; j1 <= i1; ++j1) {
            double sum2 =
                column_product_sum(coefficients, input.mdim, i1, j1, nl2, nu2);
            sum2 *= input.full_occupancy;
            const double sum1 =
                column_product_sum(coefficients, input.mdim, i1, j1, nl1, nu1);

            const std::size_t idx = packed_index(i1 - 1, j1 - 1);
            p[idx] = (sum2 + sum1 * frac) * sign;
            if (i1 == j1) {
                p[idx] += diagonal_constant;
            }
        }
    }
    return p;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
