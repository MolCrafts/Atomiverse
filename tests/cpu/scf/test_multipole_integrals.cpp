/// @file tests/cpu/scf/test_multipole_integrals.cpp
/// @brief Unit: MultipoleIntegrals closed forms, translation identity,
///        symmetry, and a p-shell second-moment analytic value.

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/cpu/scf/internal/one_electron.h"

using ATV::cpu::scf::MultipoleIntegrals;
using ATV::cpu::scf::native::Shell;

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Two single-primitive shells with raw (unnormalized) coefficients.
struct TwoShells {
    double a_alpha[1], a_coeff[1], b_alpha[1], b_coeff[1];
    Shell sh[2];
    TwoShells(double a, double b, double bx, int la = 0, int lb = 0,
              int carta = 0, int cartb = 0) {
        a_alpha[0] = a;
        a_coeff[0] = 1.0;
        b_alpha[0] = b;
        b_coeff[0] = 1.0;
        sh[0] = Shell{1, a_alpha, a_coeff, {0.0, 0.0, 0.0}, la, carta};
        sh[1] = Shell{1, b_alpha, b_coeff, {bx, 0.0, 0.0}, lb, cartb};
    }
};

} // namespace

/// ⟨s_A|(x−Cx)|s_B⟩ = S_AB · (Px − Cx) for single-primitive s functions.
TEST(MultipoleIntegrals, TwoSGaussiansClosedForm) {
    const double a = 1.0, b = 2.0, bx = 1.0;
    TwoShells ts{a, b, bx};
    const double origin[3] = {0.3, 0.0, 0.0};
    const MultipoleIntegrals m{ts.sh, 2, origin};

    const double p = a + b;
    const double S = std::pow(kPi / p, 1.5) * std::exp(-a * b / p * bx * bx);
    const double Px = b * bx / p;
    EXPECT_NEAR(m.dipole(0)(0, 1), S * (Px - origin[0]), 1e-14);
    EXPECT_NEAR(m.dipole(1)(0, 1), 0.0, 1e-15);
    EXPECT_NEAR(m.dipole(2)(0, 1), 0.0, 1e-15);
}

/// Translation identity M¹x(C′) = M¹x(C) + (Cx − C′x)·S, with the analytic
/// s|s overlap as the S reference.
TEST(MultipoleIntegrals, TranslationIdentity) {
    const double a = 0.7, b = 1.3, bx = 0.9;
    TwoShells ts{a, b, bx};
    const double c1[3] = {0.0, 0.0, 0.0};
    const double c2[3] = {0.5, -0.2, 0.1};
    const MultipoleIntegrals m1{ts.sh, 2, c1};
    const MultipoleIntegrals m2{ts.sh, 2, c2};

    const double p = a + b;
    const double S = std::pow(kPi / p, 1.5) * std::exp(-a * b / p * bx * bx);
    for (int axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                const double s_ij =
                    (i == j) ? std::pow(kPi / (2.0 * (i ? b : a)), 1.5) : S;
                EXPECT_NEAR(
                    m2.dipole(axis)(i, j),
                    m1.dipole(axis)(i, j) + (c1[axis] - c2[axis]) * s_ij, 1e-13)
                    << "axis " << axis << " (" << i << "," << j << ")";
            }
        }
    }
}

/// All nine operator matrices are symmetric.
TEST(MultipoleIntegrals, Symmetry) {
    TwoShells ts{0.8, 1.1, 0.6, /*la=*/1, /*lb=*/0, /*carta=*/0};
    const double origin[3] = {0.2, 0.1, -0.3};
    const MultipoleIntegrals m{ts.sh, 2, origin};
    for (int a = 0; a < 3; ++a)
        EXPECT_NEAR(m.dipole(a)(0, 1), m.dipole(a)(1, 0), 1e-15);
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(m.second(k)(0, 1), m.second(k)(1, 0), 1e-15);
}

/// p-shell second moment: ⟨p_x|(x−0)²|p_x⟩ for a unit-coefficient primitive
/// p_x at the origin with α = 1: ∫x⁴e^{−2x²}dx·(∫e^{−2y²}dy)² =
/// (3/16)√(π/2)·(π/2) = (3/16)(π/2)^{3/2}.
TEST(MultipoleIntegrals, PShellSecondMomentAnalytic) {
    double alpha[1] = {1.0}, coeff[1] = {1.0};
    Shell px{1, alpha, coeff, {0.0, 0.0, 0.0}, /*l=*/1, /*cart=*/0};
    const double origin[3] = {0.0, 0.0, 0.0};
    const MultipoleIntegrals m{&px, 1, origin};
    const double want = 3.0 / 16.0 * std::pow(kPi / 2.0, 1.5);
    EXPECT_NEAR(m.second(0)(0, 0), want, 1e-14); // xx
    // yy on p_x: ⟨x|y²|x⟩ = ∫x²e^{−2x²}·∫y²e^{−2y²}·∫e^{−2z²}
    //           = (1/4)√(π/2)·(1/4)√(π/2)·√(π/2) = (1/16)(π/2)^{3/2}.
    EXPECT_NEAR(m.second(3)(0, 0), 1.0 / 16.0 * std::pow(kPi / 2.0, 1.5),
                1e-14);
    EXPECT_NEAR(m.second(1)(0, 0), 0.0, 1e-15); // xy odd
}
