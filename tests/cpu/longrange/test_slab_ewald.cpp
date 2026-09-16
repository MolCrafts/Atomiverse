/// @file tests/cpu/longrange/test_slab_ewald.cpp
/// @brief Unit gates for KSpaceEwald slab_factor (Yeh–Berkowitz / LAMMPS slab).
///
/// Unit-only: pure math of slab_corr + small-N force/energy consistency.
/// Full energy vs LAMMPS offline pin lives in regressions/md/slab-2d-ewald-foothold.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/longrange/slab_corr.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

using namespace ATV;

namespace {

// Parallel-plate toy: 4×(+1) at z=5, 4×(−1) at z=15; box 12×12×20.
cpu::Frame make_plate_slab() {
    cpu::Frame frame(8);
    const double xy[4][2] = {{3, 3}, {9, 3}, {3, 9}, {9, 9}};
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = xy[i][0];
        frame.span(kAtom_y)[i] = xy[i][1];
        frame.span(kAtom_z)[i] = 5.0;
        frame.span(kAtom_charge)[i] = 1.0;
        frame.span(kAtom_mass)[i] = 1.0;
        frame.span(kAtom_type)[i] = 0;
        frame.span(kAtom_tag)[i] = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i + 4] = xy[i][0];
        frame.span(kAtom_y)[i + 4] = xy[i][1];
        frame.span(kAtom_z)[i + 4] = 15.0;
        frame.span(kAtom_charge)[i + 4] = -1.0;
        frame.span(kAtom_mass)[i + 4] = 1.0;
        frame.span(kAtom_type)[i + 4] = 1;
        frame.span(kAtom_tag)[i + 4] = i + 5;
    }
    frame.box = cpu::Box::orthorhombic(12.0, 12.0, 20.0);
    return frame;
}

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

double eval_energy(cpu::Frame& f, const cpu::md::PairList& pairs, double alpha,
                   double cutoff, double accuracy, double slab_factor) {
    for (int i = 0; i < f.nlocal(); ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyPair) = 0;
    for (int s = 0; s < 6; ++s)
        f.meta.get(kStress)[s] = 0;

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::BuildContext bctx{ctx, f};
    cpu::LaunchContext lctx{ctx, f, 0, 0};

    cpu::CoulLong real(alpha, cutoff);
    cpu::KSpaceEwald recip(alpha, accuracy);
    recip.slab_factor = slab_factor;
    real.build(bctx);
    recip.build(bctx);
    real.launch(lctx);
    recip.launch(lctx);
    real.destroy({ctx, f});
    recip.destroy({ctx, f});
    return f.meta.get(kEnergy);
}

} // namespace

TEST(SlabCorrMath, NeutralDipoleFormula) {
    // Mz = 4*(+1)*5 + 4*(-1)*15 = −40; V' = 12*12*20*3 = 8640
    const double mz = -40.0;
    const double vol = 12.0 * 12.0 * 20.0 * 3.0;
    const double zprd = 20.0 * 3.0;
    const double e =
        longrange::slab_corr_energy(mz, /*Q=*/0.0, /*mz_r2=*/0.0, vol, zprd);
    const double expect = units::coulomb * 2.0 * units::pi * mz * mz / vol;
    EXPECT_NEAR(e, expect, 1e-12);
}

TEST(SlabCorrMath, ForceMatchesEnergyGradient) {
    // Single movable +1 charge; fixed background −1 so Q=0, Mz depends on z.
    // E_slab = ke * 2π / V * (q1 z1 + q2 z2)²
    // dE/dz1 = ke * 4π / V * Mz * q1  ⇒  Fz1 = −dE/dz1 = −ke 4π Mz q1 / V
    const double vol = 1000.0;
    const double mz = 2.0 * 3.0 + (-1.0) * 1.0; // = 5
    double fz[1] = {0.0};
    StorageReal q[1] = {2.0};
    StorageReal z[1] = {3.0};
    longrange::slab_corr_add_forces(q, z, 1, mz, /*Q=*/1.0, vol, fz);
    // ffact * q * (Mz − Q z) = ke*(-4π/V)*2*(5 − 1*3) = ke*(-4π/V)*2*2
    const double expect =
        units::coulomb * (-4.0 * units::pi / vol) * 2.0 * (mz - 1.0 * 3.0);
    EXPECT_NEAR(fz[0], expect, 1e-12);
}

TEST(SlabEwald, ZeroSlabFactorIs3D) {
    auto frame = make_plate_slab();
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);
    const double cutoff = 5.9;
    const double alpha = 0.5;
    const double e0 =
        eval_energy(frame, pairs, alpha, cutoff, 1e-6, /*slab=*/0.0);
    // Pure 3D path must still run (finite energy).
    EXPECT_TRUE(std::isfinite(e0));
    EXPECT_NE(e0, 0.0);
}

TEST(SlabEwald, SlabChangesEnergy) {
    auto frame = make_plate_slab();
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);
    const double cutoff = 5.9;
    const double alpha = 0.5;
    const double e3d =
        eval_energy(frame, pairs, alpha, cutoff, 1e-6, /*slab=*/0.0);
    const double eslab =
        eval_energy(frame, pairs, alpha, cutoff, 1e-6, /*slab=*/3.0);
    // Quasi-2D correction must move the energy for a nonzero Mz system.
    EXPECT_GT(std::fabs(eslab - e3d), 1.0);
    EXPECT_TRUE(std::isfinite(eslab));
}

TEST(SlabEwald, FiniteDifferenceForcesSlab) {
    auto frame = make_plate_slab();
    // Break xy symmetry slightly so Fx/Fy are not all zero by construction.
    frame.span(kAtom_x)[0] = 3.1;
    frame.span(kAtom_y)[1] = 3.2;
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);
    const double cutoff = 5.9;
    const double alpha = 0.5;
    const double acc = 1e-8;
    const double slab = 3.0;
    const double h = 1e-5;

    auto force_comp = [&](int a, int axis) {
        eval_energy(frame, pairs, alpha, cutoff, acc, slab);
        return axis == 0   ? frame.span(kAtom_fx)[a]
               : axis == 1 ? frame.span(kAtom_fy)[a]
                           : frame.span(kAtom_fz)[a];
    };
    auto energy_at = [&](int a, int axis, double delta) {
        auto pos = axis == 0   ? frame.span(kAtom_x)
                   : axis == 1 ? frame.span(kAtom_y)
                               : frame.span(kAtom_z);
        const double orig = pos[a];
        pos[a] = orig + delta;
        const double e = eval_energy(frame, pairs, alpha, cutoff, acc, slab);
        pos[a] = orig;
        return e;
    };

    for (int a = 0; a < 8; ++a)
        for (int axis = 0; axis < 3; ++axis) {
            const double f_a = force_comp(a, axis);
            const double grad =
                (energy_at(a, axis, h) - energy_at(a, axis, -h)) / (2 * h);
            EXPECT_NEAR(f_a, -grad, 5.0e-4) << "atom " << a << " axis " << axis;
        }
}

TEST(SlabEwald, InvalidSlabFactorThrows) {
    cpu::KSpaceEwald e(0.5, 1e-5);
    e.slab_factor = 0.5; // 0 < f <= 1 is illegal
    auto frame = make_plate_slab();
    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, frame};
    EXPECT_THROW(e.build(bctx), std::invalid_argument);
}
