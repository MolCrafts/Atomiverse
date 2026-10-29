/// @file tests/cpu/potentials/test_dihedral_fourier.cpp
/// @brief DihedralFourier (shared PeriodicTorsionSet): energy, forces, terms.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/dihedral_fourier.h"

using namespace ATV;

namespace {

/// Generic non-planar 4-atom chain (φ not a special angle).
cpu::Frame make_chain() {
    cpu::Frame frame(4);
    const double xyz[4][3] = {
        {0.1, -0.2, 0.3}, {1.4, 0.1, 0.2}, {1.9, 1.5, -0.1}, {3.1, 1.8, 0.9}};
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = xyz[i][0];
        frame.span(kAtom_y)[i] = xyz[i][1];
        frame.span(kAtom_z)[i] = xyz[i][2];
        frame.span(kAtom_mass)[i] = 12.0;
        frame.span(kAtom_type)[i] = 0;
    }
    frame.box = cpu::Box::cubic(40.0);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    frame.add_dihedral(0, 1, 2, 3, 0);
    return frame;
}

template <typename Pot>
double eval_torsion(cpu::Frame& frame, Pot& pot) {
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_fx)[i] = 0;
        frame.span(kAtom_fy)[i] = 0;
        frame.span(kAtom_fz)[i] = 0;
    }
    frame.meta.get(kEnergy) = 0;
    frame.meta.get(kEnergyDihedral) = 0;
    cpu::Context ctx;
    pot.build({ctx, frame});
    pot.launch({ctx, frame, 0, 0});
    pot.destroy({ctx, frame});
    return frame.meta.get(kEnergy);
}

double dihedral_phi(const cpu::Frame& f) {
    auto p = [&](int i) {
        return std::vector<double>{f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                                   f.span(kAtom_z)[i]};
    };
    auto sub = [](auto a, auto b) {
        return std::vector<double>{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto cross = [](auto a, auto b) {
        return std::vector<double>{a[1] * b[2] - a[2] * b[1],
                                   a[2] * b[0] - a[0] * b[2],
                                   a[0] * b[1] - a[1] * b[0]};
    };
    auto dot = [](auto a, auto b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    const auto b1 = sub(p(1), p(0)), b2 = sub(p(2), p(1)), b3 = sub(p(3), p(2));
    const auto n1 = cross(b1, b2), n2 = cross(b2, b3);
    const auto m = cross(n1, n2);
    const double b2n = std::sqrt(dot(b2, b2));
    return std::atan2(dot(m, b2) / b2n, dot(n1, n2));
}

const std::vector<std::vector<cpu::TorsionTerm>> kTerms = {
    {{0.035, 1, M_PI}, {0.017, 2, 0.0}, {0.041, 3, 0.7}}};

} // namespace

TEST(DihedralFourier, EnergyIsSumOfTerms) {
    auto frame = make_chain();
    const double phi = dihedral_phi(frame);
    double expect = 0.0;
    for (const auto& t : kTerms[0])
        expect += t.K * (1.0 + std::cos(t.n * phi - t.delta));
    cpu::DihedralFourier pot(kTerms);
    EXPECT_NEAR(eval_torsion(frame, pot), expect, 1e-14);
    EXPECT_NEAR(frame.meta.get(kEnergyDihedral), expect, 1e-14);
}

TEST(DihedralFourier, OneTermEqualsCharmm) {
    auto f1 = make_chain();
    auto f2 = make_chain();
    cpu::DihedralFourier fourier({{{0.04, 3, 0.4}}});
    cpu::DihedralCharmm charmm({0.04}, {3}, {0.4});
    EXPECT_DOUBLE_EQ(eval_torsion(f1, fourier), eval_torsion(f2, charmm));
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(f1.span(kAtom_fx)[i], f2.span(kAtom_fx)[i]);
}

TEST(DihedralFourier, ForcesMatchFiniteDifference) {
    cpu::DihedralFourier pot(kTerms);
    auto frame = make_chain();
    eval_torsion(frame, pot);
    const ColKey<StorageReal> pos[3] = {kAtom_x, kAtom_y, kAtom_z};
    const ColKey<StorageReal> frc[3] = {kAtom_fx, kAtom_fy, kAtom_fz};
    constexpr double h = 1e-6;
    for (int i = 0; i < 4; ++i)
        for (int d = 0; d < 3; ++d) {
            auto plus = make_chain();
            auto minus = make_chain();
            plus.span(pos[d])[i] += h;
            minus.span(pos[d])[i] -= h;
            const double fd =
                -(eval_torsion(plus, pot) - eval_torsion(minus, pot))
                / (2.0 * h);
            EXPECT_NEAR(frame.span(frc[d])[i], fd, 1e-8)
                << "atom " << i << " axis " << d;
        }
}

TEST(DihedralFourier, AllZeroTypeContributesNothing) {
    auto frame = make_chain();
    // Name the table. `{{}}` is ambiguous on GCC 11: it also matches the
    // implicit copy and move constructors.
    std::vector<std::vector<cpu::TorsionTerm>> terms(1);
    cpu::DihedralFourier pot(terms);
    EXPECT_EQ(eval_torsion(frame, pot), 0.0);
    EXPECT_EQ(frame.span(kAtom_fx)[0], 0.0);
}

TEST(DihedralFourier, RejectsBadParams) {
    EXPECT_THROW(cpu::DihedralFourier({}), std::invalid_argument);
    EXPECT_THROW(cpu::DihedralFourier({{{1.0, 7, 0.0}}}),
                 std::invalid_argument);
    EXPECT_THROW(cpu::DihedralFourier({{{NAN, 1, 0.0}}}),
                 std::invalid_argument);
}

TEST(DihedralFourier, MissingTypeThrows) {
    auto frame = make_chain();
    frame.span(kAtom_dihedral_type)(1, 0) = 3; // type beyond the table
    cpu::DihedralFourier pot(kTerms);
    EXPECT_THROW(eval_torsion(frame, pot), std::runtime_error);
}
