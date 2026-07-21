/// @file tests/cpu/gpw/test_cpu_gpw_hartree_grid.cpp
/// @brief GPW Hartree assembly: E_H = 1/2 Tr(P J) self-consistency, symmetry.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/hartree_grid.h"

using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;
using ATV::cpu::gpw::hartree_grid;
using ATV::cpu::gpw::HartreeOnGrid;
using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::UniformGrid;

namespace {
GridBasisFunction s_function(double alpha, const double c[3]) {
    GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = c[0];
    p.center[1] = c[1];
    p.center[2] = c[2];
    p.coeff = 1.0;
    return {p};
}
} // namespace

// J_uv = integral phi_u V_H phi_v, with V_H from rho[P]. By construction the
// Hartree energy satisfies E_H = 1/2 integral rho V_H = 1/2 Tr(P J).
TEST(GpwHartree, EnergyEqualsHalfTracePJ) {
    const double cell[3] = {12.0, 12.0, 12.0};
    UniformGrid grid = make_uniform_grid(cell, 300.0);

    const double cA[3] = {5.5, 6.0, 6.0};
    const double cB[3] = {6.5, 6.0, 6.0};
    std::vector<GridBasisFunction> basis = {s_function(0.9, cA),
                                            s_function(1.1, cB)};
    std::vector<double> P = {1.0, 0.2, 0.2, 0.8};

    HartreeOnGrid h = hartree_grid(grid, basis, P);

    // Tr(P J) = sum_uv P_uv J_uv.
    double trpj = 0.0;
    for (int u = 0; u < 2; ++u)
        for (int v = 0; v < 2; ++v)
            trpj += P[u * 2 + v] * h.j_matrix[u * 2 + v];

    EXPECT_NEAR(h.e_hartree, 0.5 * trpj,
                1e-10 * std::max(1.0, std::abs(h.e_hartree)));
    EXPECT_GT(h.e_hartree, 0.0); // Hartree self-repulsion is positive
    // J symmetric.
    EXPECT_NEAR(h.j_matrix[1], h.j_matrix[2],
                1e-12 * std::max(1.0, std::abs(h.j_matrix[1])));
}
