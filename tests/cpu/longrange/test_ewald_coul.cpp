/// @file tests/cpu/longrange/test_ewald_coul.cpp
/// @brief Validation of the CPU Ewald electrostatics (CoulLong + KSpaceEwald).
///
/// Three checks, none requiring external references:
///   1. Madelung constant — the Ewald energy of a rock-salt (NaCl) lattice
///      under PBC equals −½·N·M·k_C/d with M = 1.747565 (the classic
///      electrostatic test).
///   2. α-independence — the total electrostatic energy is invariant to the
///      Ewald splitting parameter, the defining property of the method.
///   3. Finite-difference forces — analytic force = −dE/dx for the combined
///      real-space + reciprocal-space evaluator.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

#include "periodic_rocksalt.h"

using namespace ATV;
using ATV::test::build_dense_halflist;
using ATV::test::electrostatic_energy;
using ATV::test::kMadelungNaCl;
using ATV::test::PeriodicRockSalt;

namespace {} // namespace

TEST(EwaldCoul, MadelungNaCl) {
    const double d = 2.82; // nearest-neighbor spacing [Å]
    const int n = 4; // 64 ions; images across the boundary are explicit rows
    const double cutoff = 1.9 * d;
    PeriodicRockSalt rs(d, n, cutoff);

    const double alpha = 5.0 / cutoff; // erfc(α·cutoff) ≈ 1e-12
    const double e = electrostatic_energy(rs.frame, rs.context, alpha, cutoff,
                                          cpu::KSpaceEwald(alpha, 1e-7));

    const int N = n * n * n;
    const double e_expected =
        -0.5 * N * kMadelungNaCl * ATV::units::UnitSystem::lj().qqrd2e() / d;
    EXPECT_NEAR(e / e_expected, 1.0, 1e-4)
        << "E=" << e << "  expected=" << e_expected;
}

TEST(EwaldCoul, AlphaIndependence) {
    const double d = 2.82;
    const int n = 4;
    const double cutoff = 1.9 * d;
    PeriodicRockSalt rs(d, n, cutoff);

    const double e1 =
        electrostatic_energy(rs.frame, rs.context, 4.0 / cutoff, cutoff,
                             cpu::KSpaceEwald(4.0 / cutoff, 1e-8));
    const double e2 =
        electrostatic_energy(rs.frame, rs.context, 6.0 / cutoff, cutoff,
                             cpu::KSpaceEwald(6.0 / cutoff, 1e-8));
    EXPECT_NEAR(e1, e2, 1e-3) << "e1=" << e1 << " e2=" << e2;
}

TEST(EwaldCoul, FiniteDifferenceForces) {
    // Small asymmetric set of charges in a box so every component is nonzero.
    cpu::Frame frame(4);
    const double coords[4][3] = {
        {1.0, 1.0, 1.0}, {3.2, 1.4, 1.1}, {1.3, 3.1, 2.4}, {2.7, 2.2, 3.6}};
    const double q[4] = {0.6, -0.4, 0.5, -0.7};
    for (int i = 0; i < 4; ++i) {
        frame.span(ATV::kAtom_x)[i] = coords[i][0];
        frame.span(ATV::kAtom_y)[i] = coords[i][1];
        frame.span(ATV::kAtom_z)[i] = coords[i][2];
        frame.span(ATV::kAtom_charge)[i] = q[i];
        frame.span(ATV::kAtom_mass)[i] = 1.0;
        frame.span(ATV::kAtom_type)[i] = 0;
        frame.span(ATV::kAtom_tag)[i] = i + 1;
    }
    // Every pair is nearer than any periodic image (max separation 3.33 Å,
    // nearest image > 5.4 Å), so the dense owner-only list is complete.
    frame.box = ATV::cpu::Box::cubic(8.0);
    cpu::md::PairList pairs;
    build_dense_halflist(frame, pairs);
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;

    const double cutoff = 3.9;
    const double alpha = 0.45;
    const double acc = 1e-8;

    auto force = [&](int a, int axis) {
        electrostatic_energy(frame, ctx, alpha, cutoff,
                             cpu::KSpaceEwald(alpha, acc));
        return axis == 0   ? frame.span(ATV::kAtom_fx)[a]
               : axis == 1 ? frame.span(ATV::kAtom_fy)[a]
                           : frame.span(ATV::kAtom_fz)[a];
    };
    auto energy_at = [&](int a, int axis, double delta) {
        auto pos = axis == 0   ? frame.span(ATV::kAtom_x)
                   : axis == 1 ? frame.span(ATV::kAtom_y)
                               : frame.span(ATV::kAtom_z);
        const double orig = pos[a];
        pos[a] = orig + delta;
        const double e = electrostatic_energy(frame, ctx, alpha, cutoff,
                                              cpu::KSpaceEwald(alpha, acc));
        pos[a] = orig;
        return e;
    };

    const double h = 1.0e-5;
    for (int a = 0; a < 4; ++a)
        for (int axis = 0; axis < 3; ++axis) {
            const double f_analytic = force(a, axis);
            const double grad =
                (energy_at(a, axis, h) - energy_at(a, axis, -h)) / (2 * h);
            EXPECT_NEAR(f_analytic, -grad, 1.0e-4)
                << "atom " << a << " axis " << axis;
        }
}
