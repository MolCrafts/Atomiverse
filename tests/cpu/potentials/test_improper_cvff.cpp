/// @file tests/cpu/potentials/test_improper_cvff.cpp
/// @brief ImproperCvff: V = K (1 + d cos(n φ)) over the improper columns.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/improper_cvff.h"

using namespace ATV;

namespace {

/// Planar-ish trigonal centre (atom 2 is central in Amber order i-j-k-l).
cpu::Frame make_improper() {
    cpu::Frame frame(4);
    const double xyz[4][3] = {{1.2, 0.1, 0.0},
                              {-0.6, 1.1, 0.1},
                              {0.0, 0.0, 0.25},
                              {-0.6, -1.0, -0.1}};
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = xyz[i][0] + 5.0;
        frame.span(kAtom_y)[i] = xyz[i][1] + 5.0;
        frame.span(kAtom_z)[i] = xyz[i][2] + 5.0;
        frame.span(kAtom_mass)[i] = 12.0;
        frame.span(kAtom_type)[i] = 0;
    }
    frame.box = cpu::Box::cubic(30.0);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    frame.add_improper(0, 1, 2, 3, 0);
    return frame;
}

double eval_torsion(cpu::Frame& frame, cpu::ImproperCvff& pot) {
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_fx)[i] = 0;
        frame.span(kAtom_fy)[i] = 0;
        frame.span(kAtom_fz)[i] = 0;
    }
    frame.meta.get(kEnergy) = 0;
    frame.meta.get(kEnergyImproper) = 0;
    cpu::Context ctx;
    pot.build({ctx, frame});
    pot.launch({ctx, frame, 0, 0});
    pot.destroy({ctx, frame});
    return frame.meta.get(kEnergy);
}

double cos_phi(const cpu::Frame& f) {
    double p[4][3];
    for (int i = 0; i < 4; ++i) {
        p[i][0] = f.span(kAtom_x)[i];
        p[i][1] = f.span(kAtom_y)[i];
        p[i][2] = f.span(kAtom_z)[i];
    }
    double b1[3], b2[3], b3[3], n1[3], n2[3];
    for (int d = 0; d < 3; ++d) {
        b1[d] = p[1][d] - p[0][d];
        b2[d] = p[2][d] - p[1][d];
        b3[d] = p[3][d] - p[2][d];
    }
    auto cross = [](const double* a, const double* b, double* c) {
        c[0] = a[1] * b[2] - a[2] * b[1];
        c[1] = a[2] * b[0] - a[0] * b[2];
        c[2] = a[0] * b[1] - a[1] * b[0];
    };
    cross(b1, b2, n1);
    cross(b2, b3, n2);
    const double dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
    const double l1 = std::sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]);
    const double l2 = std::sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]);
    return dot / (l1 * l2);
}

} // namespace

TEST(ImproperCvff, EnergyMatchesLammpsForm) {
    for (int d : {1, -1})
        for (int n : {0, 1, 2, 3}) {
            auto frame = make_improper();
            const double phi = std::acos(cos_phi(frame));
            const double K = 0.0477;
            cpu::ImproperCvff pot({K}, {d}, {n});
            const double expect = K * (1.0 + d * std::cos(n * phi));
            EXPECT_NEAR(eval_torsion(frame, pot), expect, 1e-14)
                << "d=" << d << " n=" << n;
            EXPECT_NEAR(frame.meta.get(kEnergyImproper), expect, 1e-14);
            EXPECT_EQ(frame.meta.get(kEnergyDihedral), 0.0);
        }
}

TEST(ImproperCvff, ForcesMatchFiniteDifference) {
    cpu::ImproperCvff pot({0.455}, {-1}, {2});
    auto frame = make_improper();
    eval_torsion(frame, pot);
    const ColKey<StorageReal> pos[3] = {kAtom_x, kAtom_y, kAtom_z};
    const ColKey<StorageReal> frc[3] = {kAtom_fx, kAtom_fy, kAtom_fz};
    constexpr double h = 1e-6;
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 4; ++i)
        for (int d = 0; d < 3; ++d) {
            auto plus = make_improper();
            auto minus = make_improper();
            plus.span(pos[d])[i] += h;
            minus.span(pos[d])[i] -= h;
            const double fd =
                -(eval_torsion(plus, pot) - eval_torsion(minus, pot))
                / (2.0 * h);
            EXPECT_NEAR(frame.span(frc[d])[i], fd, 1e-8)
                << "atom " << i << " axis " << d;
            net[d] += frame.span(frc[d])[i];
        }
    for (double v : net)
        EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(ImproperCvff, RejectsBadParams) {
    EXPECT_THROW(cpu::ImproperCvff({}, {}, {}), std::invalid_argument);
    EXPECT_THROW(cpu::ImproperCvff({1.0}, {0}, {2}), std::invalid_argument);
    EXPECT_THROW(cpu::ImproperCvff({1.0}, {1}, {7}), std::invalid_argument);
    EXPECT_THROW(cpu::ImproperCvff({1.0, 2.0}, {1}, {2}),
                 std::invalid_argument);
}
