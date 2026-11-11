#include "atomiverse/cpu/scf/internal/orbital_canonicalization.h"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

namespace {
using namespace ATV::cpu::scf;
using ATV::cpu::scf::internal::semicanonicalize;

TEST(OrbitalCanonicalization, PreservesMetricDensityAndPhysicalPulayWeight) {
    RealMatrix F = {{-1.2, 0.3, 0.08, 0.02},
                    {0.3, -0.7, -0.04, 0.06},
                    {0.08, -0.04, 0.4, 0.2},
                    {0.02, 0.06, 0.2, 0.9}};
    RealMatrix S = {{2, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 1.5}};
    RealMatrix C = xt::zeros<double>({4, 4});
    for (int i = 0; i < 4; ++i)
        C(i, i) = 1 / std::sqrt(S(i, i));
    const RealMatrix occ = xt::view(C, xt::all(), xt::range(0, 2));
    const RealMatrix P = xt::linalg::dot(occ, xt::transpose(occ));
    const RealMatrix W = xt::linalg::dot(P, RealMatrix(xt::linalg::dot(F, P)));
    RealVector energies;
    const RealMatrix ai = semicanonicalize(F, C, energies, 2);
    const RealMatrix rotated_occ = xt::view(C, xt::all(), xt::range(0, 2));
    const RealMatrix after =
        xt::linalg::dot(rotated_occ, xt::transpose(rotated_occ));
    const RealMatrix metric =
        xt::linalg::dot(xt::transpose(C), RealMatrix(xt::linalg::dot(S, C)));
    const RealMatrix mo =
        xt::linalg::dot(xt::transpose(C), RealMatrix(xt::linalg::dot(F, C)));
    RealMatrix from_orbitals = xt::zeros<double>({4, 4});
    for (int i = 0; i < 2; ++i)
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                from_orbitals(a, b) += energies(i) * C(a, i) * C(b, i);
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            EXPECT_NEAR(after(a, b), P(a, b), 1e-12);
            EXPECT_NEAR(metric(a, b), a == b ? 1 : 0, 1e-12);
            EXPECT_NEAR(from_orbitals(a, b), W(a, b), 1e-12);
            if (a != b && (a < 2) == (b < 2))
                EXPECT_NEAR(mo(a, b), 0, 1e-12);
        }
    for (int a = 0; a < 4; ++a)
        EXPECT_NEAR(energies(a), mo(a, a), 1e-12);
    for (int a = 0; a < 2; ++a)
        for (int i = 0; i < 2; ++i)
            EXPECT_NEAR(ai(a, i), mo(a + 2, i), 1e-12);
}

TEST(OrbitalCanonicalization, EmptyOccupiedOrVirtualSpacesAreValid) {
    const RealMatrix F = {{-0.8, 0.1}, {0.1, 0.5}};
    for (int occupied : {0, 2}) {
        RealMatrix C = xt::eye<double>(2);
        RealVector energies;
        const auto ai = semicanonicalize(F, C, energies, occupied);
        EXPECT_EQ(ai.size(), 0u);
        EXPECT_EQ(ai.shape()[0], std::size_t(2 - occupied));
        EXPECT_EQ(ai.shape()[1], std::size_t(occupied));
        EXPECT_EQ(energies.size(), 2u);
    }
}

TEST(OrbitalCanonicalization, InvalidOccupationRejectsBeforeChangingOrbitals) {
    const RealMatrix F = {{-0.8, 0.1}, {0.1, 0.5}};
    RealMatrix C = xt::eye<double>(2);
    RealVector energies = {42, 43};
    EXPECT_THROW(semicanonicalize(F, C, energies, 3), std::invalid_argument);
    EXPECT_EQ(C(0, 0), 1);
    EXPECT_EQ(C(0, 1), 0);
    EXPECT_EQ(energies(0), 42);
}
} // namespace
