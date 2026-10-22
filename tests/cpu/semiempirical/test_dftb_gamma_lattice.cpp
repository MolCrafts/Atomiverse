/// @file tests/cpu/semiempirical/test_dftb_gamma_lattice.cpp
/// @brief GammaLattice on a skew cell: the Ewald split must not change γ.
///
/// The real-space image walk and the reciprocal sum have to describe the same
/// lattice (lattice vectors = cell columns). If they disagree, γ depends on
/// the splitting parameter α, which a correct Ewald sum never does.
#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/semiempirical/internal/dftb_gamma_lattice.h"

using ATV::cpu::semiempirical::internal::GammaLattice;

TEST(DftbGammaLattice, SkewCellGammaIndependentOfEwaldSplit) {
    ATV::Mat3 cell = ATV::Mat3::diagonal(7.0, 6.5, 8.0);
    cell(0, 1) = 2.4;  // xy (LAMMPS-form H: lattice vectors are columns)
    cell(0, 2) = -1.3; // xz
    cell(1, 2) = 1.9;  // yz
    ATV::cpu::Box box;
    box.set_cell(cell);
    constexpr int n = 4;
    const double u[n] = {11.0, 9.5, 12.5, 10.0};
    const double x[n] = {0.3, 2.9, 5.1, 1.7};
    const double y[n] = {0.4, 1.8, 4.6, 3.9};
    const double z[n] = {0.2, 3.3, 1.1, 6.4};
    const GammaLattice short_split(n, u, x, y, z, box, true, 6.0, 1e-10);
    const GammaLattice long_split(n, u, x, y, z, box, true, 11.0, 1e-10);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            EXPECT_NEAR(short_split.pair(a, b), long_split.pair(a, b), 1e-7)
                << "gamma(" << a << "," << b << ")";
}
