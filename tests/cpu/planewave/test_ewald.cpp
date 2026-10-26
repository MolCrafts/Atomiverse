/// @file test_ewald.cpp
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/planewave/internal/dfpt.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

TEST(Ewald, SiTwoAtoms) {
    cpu::Box box = cpu::Box::cubic(10.26); // Å (Si-scale cell)
    std::vector<Vec3> pos = {{0, 0, 0}, {2.565, 2.565, 2.565}};
    std::vector<AccReal> charges = {4, 4};
    EwaldSummation e1(pos, charges, box);
    for (auto& r : pos)
        r = r * AccReal(2);
    EwaldSummation e2(pos, charges, cpu::Box(box.cell() * AccReal(2)));
    // Coulomb energy, including the uniform background, scales as 1/L.
    AccReal diff = std::abs(e1.total_energy() - AccReal(2) * e2.total_energy());
    EXPECT_LT(static_cast<double>(diff), 1e-4); // Hartree
}

TEST(Ewald, ZeroCharges) {
    cpu::Box box = cpu::Box::cubic(10);
    std::vector<Vec3> pos = {{0, 0, 0}, {2, 2, 2}};
    std::vector<AccReal> charges = {0, 0};
    EwaldSummation e(pos, charges, box);
    EXPECT_NEAR(static_cast<double>(e.total_energy()), 0.0, 1e-12);
}

TEST(Ewald, ForceMatchesEnergyFd) {
    // Analytic Ewald force vs central FD of total_energy (same functional).
    cpu::Box box = cpu::Box::cubic(8.0); // Å
    std::vector<Vec3> pos = {{0.0, 0.0, 0.0}, {1.5, 0.2, -0.1}};
    std::vector<AccReal> charges = {1.0, 1.0};
    const AccReal h = AccReal(1e-4); // Å

    EwaldSummation e0(pos, charges, box);
    ASSERT_EQ(e0.forces_bohr().size(), 2u);

    const double to_ev_a = units::convert(1.0, units::Force::HartreePerBohr,
                                          units::Force::eVPerAngstrom);
    // FD of energy [Ha] w.r.t. R [Å]: F_Ha/Å = -dE/dR_Å;
    // convert to Ha/Bohr: F_Ha/Bohr = F_Ha/Å / a2b
    const double a2b = 1.8897259886;

    for (int ia = 0; ia < 2; ++ia) {
        for (int k = 0; k < 3; ++k) {
            auto pp = pos;
            auto pm = pos;
            pp[static_cast<std::size_t>(ia)][k] += h;
            pm[static_cast<std::size_t>(ia)][k] -= h;
            const double ep = static_cast<double>(
                EwaldSummation(pp, charges, box).total_energy());
            const double em = static_cast<double>(
                EwaldSummation(pm, charges, box).total_energy());
            const double f_fd_ha_per_ang =
                -(ep - em) / (2.0 * static_cast<double>(h));
            const double f_fd_ha_bohr = f_fd_ha_per_ang / a2b;
            const double f_an = static_cast<double>(
                e0.forces_bohr()[static_cast<std::size_t>(ia)][k]);
            EXPECT_NEAR(f_an, f_fd_ha_bohr, 5e-4)
                << " atom=" << ia << " axis=" << k << " analytic=" << f_an
                << " fd=" << f_fd_ha_bohr << " (eV/Å analytic would be "
                << f_an * to_ev_a << ")";
        }
    }
}

TEST(Ewald, NonSymmetricCellAndForcesRotateTogether) {
    Mat3 h = Mat3::diagonal(4.0, 5.0, 6.0);
    h(0, 1) = 1.5;
    h(1, 2) = 0.8;
    Mat3 rotation = Mat3::diagonal(1.0, 1.0, 1.0);
    const double c = std::cos(0.37), s = std::sin(0.37);
    rotation(0, 0) = c;
    rotation(0, 1) = -s;
    rotation(1, 0) = s;
    rotation(1, 1) = c;
    const std::vector<Vec3> pos = {h * Vec3{0.1, 0.2, 0.3},
                                   h * Vec3{0.4, 0.7, 0.6}};
    const std::vector<Vec3> rotated = {rotation * pos[0], rotation * pos[1]};
    const std::vector<AccReal> charges = {1.0, -1.0};
    const EwaldSummation a(pos, charges, cpu::Box(h));
    const EwaldSummation b(rotated, charges, cpu::Box(rotation * h));
    EXPECT_NEAR(a.total_energy(), b.total_energy(), 1e-11);
    for (int atom = 0; atom < 2; ++atom) {
        const auto expected = rotation * a.forces_bohr()[atom];
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(b.forces_bohr()[atom][d], expected[d], 1e-11);
    }
}

TEST(Ewald, ChargedCubicUniformBackgroundMadelung) {
    // Cubic periodic Coulomb Green function with a uniform compensating
    // background: E=-alpha_M*q^2/(2L), alpha_M=2.837297479480619...
    // Makov & Payne, Phys. Rev. B 51, 4014 (1995), cubic Madelung constant.
    // This analytic pin includes the background, independent of a PW solver.
    constexpr AccReal alpha_m = AccReal(2.837297479480619);
    constexpr AccReal a2b = AccReal(1.8897259886);
    const cpu::Box box = cpu::Box::cubic(6);
    const EwaldSummation e({{0, 0, 0}}, {1}, box);
    EXPECT_NEAR(e.total_energy(), -alpha_m / (AccReal(2) * 6 * a2b), 1e-12);
    EXPECT_LT(e.background_energy(), 0);
    EXPECT_NEAR(e.forces_bohr()[0].norm(), 0, 1e-12);
    const EwaldSummation translated({{66, -78, 114}}, {1}, box);
    EXPECT_NEAR(translated.total_energy(), e.total_energy(), 1e-12);
    EXPECT_NEAR(translated.forces_bohr()[0].norm(), 0, 1e-12);
    EXPECT_DOUBLE_EQ(e.total_energy(), e.real_energy() + e.recip_energy()
                                           + e.self_energy()
                                           + e.background_energy());
}

TEST(Ewald, NeutralRockSaltMadelung) {
    // Rock-salt conventional cube: four ion pairs, nearest distance L/2.
    // The analytic lattice constant is M_NaCl=1.7475645946331821906...
    constexpr AccReal madelung = AccReal(1.7475645946331821906);
    constexpr AccReal a2b = AccReal(1.8897259886);
    const AccReal side = 5.64;
    std::vector<Vec3> pos{{0, 0, 0},    {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0},
                          {.5, .5, .5}, {.5, 0, 0},  {0, .5, 0},  {0, 0, .5}};
    for (auto& r : pos)
        r = r * side;
    const EwaldSummation e(pos, {1, 1, 1, 1, -1, -1, -1, -1},
                           cpu::Box::cubic(side));
    EXPECT_NEAR(e.total_energy(), -8 * madelung / (side * a2b), 1e-11);
    EXPECT_DOUBLE_EQ(e.background_energy(), 0);
    for (const auto& force : e.forces_bohr())
        EXPECT_NEAR(force.norm(), 0, 1e-11);
}

TEST(Ewald, InvalidPhysicalInputsRejectBeforeIndexing) {
    const cpu::Box box = cpu::Box::cubic(6);
    const AccReal nan = std::numeric_limits<AccReal>::quiet_NaN();
    EXPECT_THROW(EwaldSummation({}, {1}, box), std::invalid_argument);
    EXPECT_THROW(EwaldSummation({{0, 0, 0}}, {nan}, box),
                 std::invalid_argument);
    EXPECT_THROW(EwaldSummation({{nan, 0, 0}}, {1}, box),
                 std::invalid_argument);
    EXPECT_THROW(EwaldSummation({{0, 0, 0}}, {1}, cpu::Box{}),
                 std::invalid_argument);
    EXPECT_THROW(EwaldSummation({{0, 0, 0}, {0, 0, 0}}, {1, -1}, box),
                 std::invalid_argument);
}

TEST(Ewald, IonicHessianDifferentiatesTheSamePrecisionForce) {
    const cpu::Box box = cpu::Box::cubic(8);
    const std::vector<Vec3> pos{{.2, .3, .4}, {1.5, .2, -.1}};
    const std::vector<AccReal> charges{1, -.7};
    const auto hessian = ewald_dynamical_matrix(pos, charges, box);
    constexpr AccReal step = AccReal(1e-4); // Angstrom.
    constexpr AccReal a2b = AccReal(1.8897259886);
    auto plus = pos, minus = pos;
    plus[1][0] += step;
    minus[1][0] -= step;
    const EwaldSummation ep(plus, charges, box), em(minus, charges, box);
    for (int atom = 0; atom < 2; ++atom)
        for (int d = 0; d < 3; ++d) {
            const AccReal derivative =
                -(ep.forces_bohr()[atom][d] - em.forces_bohr()[atom][d])
                / (AccReal(2) * step * a2b);
            EXPECT_NEAR(hessian[(3 * atom + d) * 6 + 3], derivative, 1e-8);
        }
}

TEST(Ewald, MultipleLatticeImagesPreserveEnergyForceAndGammaHessian) {
    Mat3 h = Mat3::diagonal(4, 5, 6);
    h(0, 1) = 1.5;
    h(1, 2) = .8;
    const cpu::Box box(h);
    const std::vector<Vec3> pos{h * Vec3{.11, .22, .33},
                                h * Vec3{.41, .72, .63}};
    const std::vector<AccReal> charges{1, -.7};
    const EwaldSummation reference(pos, charges, box);
    const auto hessian = ewald_dynamical_matrix(pos, charges, box);
    const Vec3 shift = h * Vec3{13, -9, 17};
    for (bool shift_both : {false, true}) {
        auto translated = pos;
        translated[1] = translated[1] + shift;
        if (shift_both)
            translated[0] = translated[0] + shift;
        const EwaldSummation e(translated, charges, box);
        EXPECT_NEAR(e.total_energy(), reference.total_energy(), 1e-12);
        const auto translated_hessian =
            ewald_dynamical_matrix(translated, charges, box);
        for (std::size_t i = 0; i < hessian.size(); ++i)
            EXPECT_NEAR(translated_hessian[i], hessian[i], 1e-11);
        for (int atom = 0; atom < 2; ++atom)
            for (int d = 0; d < 3; ++d)
                EXPECT_NEAR(e.forces_bohr()[atom][d],
                            reference.forces_bohr()[atom][d], 1e-11);
    }
}

TEST(Ewald, FiniteQHessianHasRawAtomicImageGaugeCovariance) {
    const cpu::Box box = cpu::Box::cubic(5);
    const std::vector<Vec3> pos{{.4, .5, .6}, {1.3, 2.1, 1.4}};
    const std::vector<AccReal> charges{1, -.7};
    const Vec3 q{.13, .07, -.11}, image{13, -9, 17};
    const auto reference = ewald_dynamical_matrix_q(pos, charges, box, q);
    auto translated = pos;
    translated[1] = translated[1] + box.cell() * image;
    const auto shifted = ewald_dynamical_matrix_q(translated, charges, box, q);
    const AccReal angle =
        AccReal(2) * units::pi
        * (q[0] * image[0] + q[1] * image[1] + q[2] * image[2]);
    const std::complex<AccReal> phase[2] = {{1, 0},
                                            {std::cos(angle), std::sin(angle)}};
    for (int row = 0; row < 6; ++row)
        for (int column = 0; column < 6; ++column) {
            const auto expected = phase[row / 3] * std::conj(phase[column / 3])
                                  * reference[row * 6 + column];
            EXPECT_NEAR(std::abs(shifted[row * 6 + column] - expected), 0,
                        1e-11);
        }
}
