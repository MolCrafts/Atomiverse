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

using namespace ATV;

namespace {

constexpr double kMadelungNaCl = 1.747564594633182;

// Rock-salt lattice: n³ ions on a simple-cubic grid of spacing d, charges
// alternating (-1)^(i+j+k). Box edge = n·d. Neutral when n is even.
cpu::Frame make_rocksalt(double d, int n) {
    const int N = n * n * n;
    cpu::Frame frame(N);
    int idx = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                frame.span(ATV::kAtom_x)[idx] = i * d;
                frame.span(ATV::kAtom_y)[idx] = j * d;
                frame.span(ATV::kAtom_z)[idx] = k * d;
                frame.span(ATV::kAtom_charge)[idx] =
                    ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
                frame.span(ATV::kAtom_mass)[idx] = 1.0;
                frame.span(ATV::kAtom_type)[idx] = 0;
                frame.span(ATV::kAtom_tag)[idx] = idx + 1;
                ++idx;
            }
    frame.box = ATV::cpu::Box::cubic(n * d);
    return frame;
}

// Dense half neighbor list (every j > i). CoulLong applies the cutoff and
// minimum-image internally; valid while cutoff < L/2.
void build_full_halflist(const cpu::Frame& f, cpu::md::PairList& pairs) {
    const int N = f.nlocal();
    pairs.inum = N;
    pairs.ilist.resize(N);
    pairs.numneigh.resize(N);
    pairs.firstneigh.resize(N);
    pairs.neighbors.clear();
    int off = 0;
    for (int i = 0; i < N; ++i) {
        pairs.ilist[i] = i;
        pairs.firstneigh[i] = off;
        for (int j = i + 1; j < N; ++j)
            pairs.neighbors.push_back(j);
        pairs.numneigh[i] = N - 1 - i;
        off += N - 1 - i;
    }
    pairs.total_pairs = off;
    pairs.has_shifts = false;
    pairs.version = 1;
}

// Total electrostatic energy: real-space CoulLong + reciprocal KSpaceEwald.
double electrostatic_energy(cpu::Frame& f, const cpu::md::PairList& pairs,
                            double alpha, double cutoff, double accuracy) {
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(ATV::kAtom_fx)[i] = f.span(ATV::kAtom_fy)[i] =
            f.span(ATV::kAtom_fz)[i] = 0;
    }
    f.meta.get(ATV::kEnergy) = 0;
    f.meta.get(ATV::kEnergyPair) = 0;
    for (int s = 0; s < 6; ++s)
        f.meta.get(ATV::kStress)[s] = 0;

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::BuildContext bctx{ctx, f};
    cpu::LaunchContext lctx{ctx, f, 0, 0};

    cpu::CoulLong real(alpha, cutoff);
    cpu::KSpaceEwald recip(alpha, accuracy);
    real.build(bctx);
    recip.build(bctx);
    real.launch(lctx);
    recip.launch(lctx);
    real.destroy({ctx, f});
    return f.meta.get(ATV::kEnergy);
}

} // namespace

TEST(EwaldCoul, MadelungNaCl) {
    const double d = 2.82; // nearest-neighbor spacing [Å]
    const int n = 4;       // 64 ions, so cutoff < L/2 = 2d is comfortable
    auto frame = make_rocksalt(d, n);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 1.9 * d;     // < L/2 = 2d
    const double alpha = 5.0 / cutoff; // erfc(α·cutoff) ≈ 1e-12
    const double e = electrostatic_energy(frame, pairs, alpha, cutoff, 1e-7);

    const int N = n * n * n;
    const double e_expected =
        -0.5 * N * kMadelungNaCl * ATV::units::coulomb / d;
    EXPECT_NEAR(e / e_expected, 1.0, 1e-4)
        << "E=" << e << "  expected=" << e_expected;
}

TEST(EwaldCoul, AlphaIndependence) {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 1.9 * d;
    const double e1 =
        electrostatic_energy(frame, pairs, 4.0 / cutoff, cutoff, 1e-8);
    const double e2 =
        electrostatic_energy(frame, pairs, 6.0 / cutoff, cutoff, 1e-8);
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
    frame.box = ATV::cpu::Box::cubic(8.0);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 3.9;
    const double alpha = 0.45;
    const double acc = 1e-8;

    auto force = [&](int a, int axis) {
        electrostatic_energy(frame, pairs, alpha, cutoff, acc);
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
        const double e = electrostatic_energy(frame, pairs, alpha, cutoff, acc);
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
