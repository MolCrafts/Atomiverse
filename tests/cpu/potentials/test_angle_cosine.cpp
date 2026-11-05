/// @file tests/cpu/potentials/test_angle_cosine.cpp
/// @brief cpu::AngleCosine: V = K(1 + cos θ), forces, virial at a few angles.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/angle_cosine.h"

using namespace ATV;

namespace {

using Vec = std::array<double, 3>;

// Centre j at a generic point, i along a tilted unit vector at r1, k at angle
// theta from it at r2, so every Cartesian component is exercised.
struct Triple {
    Vec ri, rj, rk;
};

Triple triple(double theta, double r1, double r2) {
    const Vec rj = {3.1, 2.7, 4.3};
    // Orthonormal in-plane basis u, w (not aligned with the axes).
    const Vec u = {0.6, 0.0, 0.8};
    const Vec w = {0.0, 1.0, 0.0};
    Triple t;
    t.rj = rj;
    for (int d = 0; d < 3; ++d) {
        t.ri[d] = rj[d] + r1 * u[d];
        t.rk[d] =
            rj[d] + r2 * (std::cos(theta) * u[d] + std::sin(theta) * w[d]);
    }
    return t;
}

cpu::Frame frame_of(const Triple& t, int type = 0) {
    cpu::Frame f(3);
    const Vec* r[3] = {&t.ri, &t.rj, &t.rk};
    for (int a = 0; a < 3; ++a) {
        f.span(kAtom_x)[a] = (*r[a])[0];
        f.span(kAtom_y)[a] = (*r[a])[1];
        f.span(kAtom_z)[a] = (*r[a])[2];
        f.span(kAtom_mass)[a] = 1.0;
        f.span(kAtom_tag)[a] = a;
        f.span(kAtom_fx)[a] = f.span(kAtom_fy)[a] = f.span(kAtom_fz)[a] = 0;
    }
    f.box = cpu::Box::cubic(40.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    f.add_angle(0, 1, 2, type);
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyAngle) = 0;
    f.meta.get(kStress).fill(0);
    return f;
}

double energy_at(const Triple& t, double K) {
    auto f = frame_of(t);
    cpu::Context ctx;
    cpu::AngleCosine angle({K});
    angle.build({ctx, f});
    angle.launch({ctx, f, 0, 0});
    angle.destroy({ctx, f});
    return f.meta.get(kEnergy);
}

class AngleCosineAt : public testing::TestWithParam<double> {};

} // namespace

TEST_P(AngleCosineAt, EnergyForceAndVirialMatchAnalyticAndFiniteDifference) {
    const double theta = GetParam() * units::pi / 180.0;
    const double K = 0.37, r1 = 1.1, r2 = 0.85;
    const Triple t = triple(theta, r1, r2);
    auto f = frame_of(t);
    cpu::Context ctx;
    cpu::AngleCosine angle({K});
    angle.build({ctx, f});
    angle.launch({ctx, f, 0, 0, /*forces=*/true, /*stress=*/true});
    angle.destroy({ctx, f});

    // Energy: K(1 + cos θ), into the angle channel too.
    EXPECT_NEAR(f.meta.get(kEnergy), K * (1.0 + std::cos(theta)), 1e-14);
    EXPECT_NEAR(f.meta.get(kEnergyAngle), K * (1.0 + std::cos(theta)), 1e-14);

    // Forces: central differences of the energy, atom by atom.
    const double h = 1e-6;
    Vec force[3];
    for (int a = 0; a < 3; ++a)
        for (int d = 0; d < 3; ++d) {
            Triple plus = t, minus = t;
            Vec* p = a == 0 ? &plus.ri : a == 1 ? &plus.rj : &plus.rk;
            Vec* m = a == 0 ? &minus.ri : a == 1 ? &minus.rj : &minus.rk;
            (*p)[d] += h;
            (*m)[d] -= h;
            force[a][d] = -(energy_at(plus, K) - energy_at(minus, K)) / (2 * h);
        }
    const auto fx = f.span(kAtom_fx), fy = f.span(kAtom_fy),
               fz = f.span(kAtom_fz);
    for (int a = 0; a < 3; ++a) {
        EXPECT_NEAR(fx[a], force[a][0], 1e-8) << "atom " << a;
        EXPECT_NEAR(fy[a], force[a][1], 1e-8) << "atom " << a;
        EXPECT_NEAR(fz[a], force[a][2], 1e-8) << "atom " << a;
    }
    // Newton's third law.
    EXPECT_NEAR(fx[0] + fx[1] + fx[2], 0.0, 1e-14);
    EXPECT_NEAR(fy[0] + fy[1] + fy[2], 0.0, 1e-14);
    EXPECT_NEAR(fz[0] + fz[1] + fz[2], 0.0, 1e-14);

    // Virial (LAMMPS ev_tally): d1 ⊗ F1 + d2 ⊗ F3, Voigt xx yy zz xy xz yz,
    // here from the finite-difference forces.
    Vec d1, d2;
    for (int d = 0; d < 3; ++d) {
        d1[d] = t.ri[d] - t.rj[d];
        d2[d] = t.rk[d] - t.rj[d];
    }
    const int ab[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
    const auto& stress = f.meta.get(kStress);
    for (int c = 0; c < 6; ++c) {
        const int a = ab[c][0], b = ab[c][1];
        const double expected = d1[a] * force[0][b] + d2[a] * force[2][b];
        EXPECT_NEAR(stress[c], expected, 1e-8) << "component " << c;
    }
    // A torque-free angle term has a symmetric virial: xy == yx.
    const double yx = d1[1] * fx[0] + d2[1] * fx[2];
    EXPECT_NEAR(stress[3], yx, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(Angles, AngleCosineAt,
                         testing::Values(60.0, 90.0, 120.0, 150.0));

TEST(AngleCosine, StraightChainIsTheMinimum) {
    // θ = π: V = 0 and no force (the Kremer–Grest straight chain).
    const Triple t = triple(units::pi, 1.0, 1.0);
    auto f = frame_of(t);
    cpu::Context ctx;
    cpu::AngleCosine angle({2.0});
    angle.build({ctx, f});
    angle.launch({ctx, f, 0, 0});
    angle.destroy({ctx, f});
    EXPECT_NEAR(f.meta.get(kEnergy), 0.0, 1e-14);
    for (int a = 0; a < 3; ++a) {
        EXPECT_NEAR(f.span(kAtom_fx)[a], 0.0, 1e-12);
        EXPECT_NEAR(f.span(kAtom_fy)[a], 0.0, 1e-12);
        EXPECT_NEAR(f.span(kAtom_fz)[a], 0.0, 1e-12);
    }
}

TEST(AngleCosine, RightAngleClosedForm) {
    // θ = 90°, r1 = r2 = 1: c = 0, so F1 = −K d2 and F3 = −K d1.
    const double K = 0.8;
    const Triple t = triple(units::pi / 2, 1.0, 1.0);
    auto f = frame_of(t);
    cpu::Context ctx;
    cpu::AngleCosine angle({K});
    angle.build({ctx, f});
    angle.launch({ctx, f, 0, 0});
    angle.destroy({ctx, f});
    EXPECT_NEAR(f.meta.get(kEnergy), K, 1e-14);
    EXPECT_NEAR(f.span(kAtom_fx)[0], -K * (t.rk[0] - t.rj[0]), 1e-14);
    EXPECT_NEAR(f.span(kAtom_fy)[0], -K * (t.rk[1] - t.rj[1]), 1e-14);
    EXPECT_NEAR(f.span(kAtom_fz)[2], -K * (t.ri[2] - t.rj[2]), 1e-14);
}

TEST(AngleCosine, TypeIndexesTheStiffnessAndRequestsAreIndependent) {
    const double theta = 1.2;
    const Triple t = triple(theta, 1.0, 1.3);
    auto f = frame_of(t, /*type=*/1);
    cpu::Context ctx;
    cpu::AngleCosine angle({0.1, 0.9});
    angle.build({ctx, f});
    f.meta.get(kStress).fill(5);
    angle.launch({ctx, f, 0, 0, /*forces=*/false, /*stress=*/false});
    EXPECT_NEAR(f.meta.get(kEnergy), 0.9 * (1.0 + std::cos(theta)), 1e-14);
    for (int a = 0; a < 3; ++a)
        EXPECT_EQ(f.span(kAtom_fx)[a], 0.0);
    for (int c = 0; c < 6; ++c)
        EXPECT_EQ(f.meta.get(kStress)[c], 5.0);
    angle.destroy({ctx, f});
}

TEST(AngleCosine, RejectsBadTableAndOutOfRangeType) {
    EXPECT_THROW(cpu::AngleCosine(std::vector<double>{}),
                 std::invalid_argument);
    EXPECT_THROW(cpu::AngleCosine(std::vector<double>{std::nan("")}),
                 std::invalid_argument);
    auto f = frame_of(triple(1.0, 1.0, 1.0), /*type=*/2);
    cpu::Context ctx;
    cpu::AngleCosine angle({0.5, 0.5});
    angle.build({ctx, f});
    EXPECT_THROW(angle.launch({ctx, f, 0, 0}), std::runtime_error);
    angle.destroy({ctx, f});
}
