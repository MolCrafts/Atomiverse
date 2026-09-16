/// @file tests/cpu/longrange/test_pppm_coul.cpp
/// @brief Scientific validation of true Hockney–Eastwood PPPM
///        (CoulLong + KSpacePPPM). Built only with ATV_BUILD_KSPACE.
///
/// Gates:
///   1. Madelung constant of rock-salt NaCl
///   2. Total electrostatic energy agrees with direct Ewald
///   3. Agrees with SPME (KSpacePME) on the same system
///   4. Finite-difference forces F = −∇E
///   5. α-independence of total energy (method identity)
///   6. Pure mesh-arm selector arithmetic (pppm_fft_ranks /
///      pppm_select_mesh_arm) — no MPI, no FFT, no env

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

using namespace ATV;

namespace {

constexpr double kMadelungNaCl = 1.747564594633182;

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

void zero_force_energy_stress(cpu::Frame& f) {
    for (int i = 0; i < f.nlocal(); ++i)
        f.span(ATV::kAtom_fx)[i] = f.span(ATV::kAtom_fy)[i] =
            f.span(ATV::kAtom_fz)[i] = 0;
    f.meta.get(ATV::kEnergy) = 0;
    f.meta.get(ATV::kEnergyPair) = 0;
    for (int s = 0; s < 6; ++s)
        f.meta.get(ATV::kStress)[s] = 0;
}

template <typename KSpace>
double electrostatic_energy(cpu::Frame& f, const cpu::md::PairList& pairs,
                            double alpha, double cutoff, KSpace&& recip) {
    zero_force_energy_stress(f);

    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    cpu::BuildContext bctx{ctx, f};
    cpu::LaunchContext lctx{ctx, f, 0, 0};

    cpu::CoulLong real(alpha, cutoff);
    real.build(bctx);
    recip.build(bctx);
    real.launch(lctx);
    recip.launch(lctx);
    real.destroy({ctx, f});
    recip.destroy({ctx, f});
    return f.meta.get(ATV::kEnergy);
}

} // namespace

TEST(PppmCoul, MadelungNaCl) {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 1.9 * d;
    // Moderate α + fine mesh so HE G(k) resolves the lattice well.
    const double alpha = 3.5 / cutoff;
    const double e = electrostatic_energy(
        frame, pairs, alpha, cutoff,
        cpu::KSpacePPPM(alpha, /*spacing=*/0.5, /*order=*/5));

    const int N = n * n * n;
    const double e_expected =
        -0.5 * N * kMadelungNaCl * ATV::units::coulomb / d;
    EXPECT_NEAR(e / e_expected, 1.0, 1e-3)
        << "E=" << e << " expected=" << e_expected;
}

TEST(PppmCoul, AgreesWithEwald) {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 1.9 * d;
    const double alpha = 3.5 / cutoff;
    const double e_ewald = electrostatic_energy(frame, pairs, alpha, cutoff,
                                                cpu::KSpaceEwald(alpha, 1e-8));
    const double e_pppm = electrostatic_energy(frame, pairs, alpha, cutoff,
                                               cpu::KSpacePPPM(alpha, 0.5, 5));
    EXPECT_NEAR(e_pppm / e_ewald, 1.0, 1e-3)
        << "ewald=" << e_ewald << " pppm=" << e_pppm;
}

TEST(PppmCoul, AgreesWithPme) {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 1.9 * d;
    const double alpha = 3.5 / cutoff;
    const double e_pme = electrostatic_energy(frame, pairs, alpha, cutoff,
                                              cpu::KSpacePME(alpha, 0.5, 6));
    const double e_pppm = electrostatic_energy(frame, pairs, alpha, cutoff,
                                               cpu::KSpacePPPM(alpha, 0.5, 5));
    EXPECT_NEAR(e_pppm / e_pme, 1.0, 2e-3)
        << "pme=" << e_pme << " pppm=" << e_pppm;
}

TEST(PppmCoul, AlphaIndependence) {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 1.9 * d;
    // Fine mesh so both α values are well-resolved.
    const double e1 =
        electrostatic_energy(frame, pairs, 3.0 / cutoff, cutoff,
                             cpu::KSpacePPPM(3.0 / cutoff, 0.4, 5));
    const double e2 =
        electrostatic_energy(frame, pairs, 5.0 / cutoff, cutoff,
                             cpu::KSpacePPPM(5.0 / cutoff, 0.4, 5));
    EXPECT_NEAR(e1, e2, 5e-2 * std::max(1.0, std::fabs(e1)))
        << "e1=" << e1 << " e2=" << e2;
}

TEST(PppmCoul, FiniteDifferenceForces) {
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
    auto run = [&]() {
        return electrostatic_energy(frame, pairs, alpha, cutoff,
                                    cpu::KSpacePPPM(alpha, 0.6, 5));
    };
    auto force = [&](int a, int axis) {
        run();
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
        const double e = run();
        pos[a] = orig;
        return e;
    };

    const double h = 1e-5;
    for (int a = 0; a < 4; ++a)
        for (int axis = 0; axis < 3; ++axis) {
            const double f_analytic = force(a, axis);
            const double grad =
                (energy_at(a, axis, h) - energy_at(a, axis, -h)) / (2 * h);
            EXPECT_NEAR(f_analytic, -grad, 5e-3)
                << "atom " << a << " axis " << axis << " F=" << f_analytic
                << " -dE=" << -grad;
        }
}

TEST(PppmLifecycle, IsReciprocalLongRangeParticipant) {
    cpu::KSpacePPPM pppm(0.4, 1.0, 5);
    const auto* lr = pppm.as_long_range_participant();
    ASSERT_NE(lr, nullptr);
    EXPECT_EQ(lr->long_range_contract().role,
              cpu::LongRangeRole::ReciprocalSpace);
    EXPECT_EQ(pppm.name(), "pppm");
}

// --- pure selector arithmetic: pppm_fft_ranks -------------------------------

TEST(PppmFftRanks, LargestDivisorWithinRankBudget) {
    // Budget = min(nprocs, planes_k0); the answer is the largest divisor of
    // planes_k0 that fits, so the FFTW-MPI dim-0 slab split is never ragged.
    EXPECT_EQ(cpu::pppm_fft_ranks(32, 256), 32);
    EXPECT_EQ(cpu::pppm_fft_ranks(32, 64), 32);
    EXPECT_EQ(cpu::pppm_fft_ranks(32, 16), 16);
    EXPECT_EQ(cpu::pppm_fft_ranks(34, 4), 2);
}

TEST(PppmFftRanks, MinPlanesPerRankShrinksBudget) {
    // Budget = min(nprocs, planes_k0 / min_planes_per_rank).
    EXPECT_EQ(cpu::pppm_fft_ranks(32, 256, 2), 16);
    EXPECT_EQ(cpu::pppm_fft_ranks(32, 256, 4), 8);
}

TEST(PppmFftRanks, MinPlanesPerRankLargerThanMeshGivesOne) {
    EXPECT_EQ(cpu::pppm_fft_ranks(32, 256, 33), 1);
    EXPECT_EQ(cpu::pppm_fft_ranks(32, 8, 64), 1);
}

TEST(PppmFftRanks, PrimeMeshOrSingleRankGivesOne) {
    EXPECT_EQ(cpu::pppm_fft_ranks(31, 8), 1); // 31 prime: only 1 and 31 fit
    EXPECT_EQ(cpu::pppm_fft_ranks(32, 1), 1);
    EXPECT_EQ(cpu::pppm_fft_ranks(31, 1), 1);
}

TEST(PppmFftRanks, ThrowsOnNonPositiveArguments) {
    EXPECT_THROW(cpu::pppm_fft_ranks(0, 16), std::invalid_argument);
    EXPECT_THROW(cpu::pppm_fft_ranks(-32, 16), std::invalid_argument);
    EXPECT_THROW(cpu::pppm_fft_ranks(32, 0), std::invalid_argument);
    EXPECT_THROW(cpu::pppm_fft_ranks(32, -4), std::invalid_argument);
    EXPECT_THROW(cpu::pppm_fft_ranks(32, 16, 0), std::invalid_argument);
    EXPECT_THROW(cpu::pppm_fft_ranks(32, 16, -1), std::invalid_argument);
}

TEST(PppmFftRanks, ResultDividesMeshAndFitsBudget) {
    // Same tuples, checked as the two structural invariants the FFTW-MPI slab
    // split depends on: exact divisibility, and never more ranks than planes.
    struct Row {
        int planes;
        int nprocs;
        int min_planes;
    };
    const Row rows[] = {{32, 256, 1}, {32, 64, 1},  {32, 16, 1},
                        {32, 256, 2}, {34, 4, 1},   {31, 8, 1},
                        {32, 1, 1},   {32, 256, 4}, {31, 1, 1}};
    for (const Row& row : rows) {
        const int r =
            cpu::pppm_fft_ranks(row.planes, row.nprocs, row.min_planes);
        EXPECT_GE(r, 1) << "planes=" << row.planes << " np=" << row.nprocs;
        EXPECT_EQ(row.planes % r, 0)
            << "ragged slab: planes=" << row.planes << " r=" << r;
        EXPECT_LE(r, row.nprocs) << "planes=" << row.planes;
        EXPECT_LE(r, row.planes) << "np=" << row.nprocs;
    }
}

// --- pure selector policy: pppm_select_mesh_arm -----------------------------

TEST(PppmSelectMeshArm, SingleRankIsReplicated) {
    EXPECT_EQ(cpu::pppm_select_mesh_arm(32, 1, 1),
              cpu::PppmMeshArm::Replicated);
    EXPECT_EQ(cpu::pppm_select_mesh_arm(31, 1, 1),
              cpu::PppmMeshArm::Replicated);
}

TEST(PppmSelectMeshArm, AllRanksInFftGroupIsBrickWorld) {
    EXPECT_EQ(cpu::pppm_select_mesh_arm(32, 16, 16),
              cpu::PppmMeshArm::BrickWorld);
    EXPECT_EQ(cpu::pppm_select_mesh_arm(32, 32, 32),
              cpu::PppmMeshArm::BrickWorld);
}

TEST(PppmSelectMeshArm, FewerFftRanksThanNprocsIsBrickSubgroup) {
    // Falsified-policy pin: the old `mesh <= 64^3 && ranks >= 16 -> replicated`
    // rule picked Replicated at np=64, contradicted by measurement (brick
    // 4.78 s vs replicated 5.43 s) — 32 planes on 64 ranks is a subgroup.
    EXPECT_EQ(cpu::pppm_select_mesh_arm(32, 64, 32),
              cpu::PppmMeshArm::BrickSubgroup);
    EXPECT_EQ(cpu::pppm_select_mesh_arm(32, 256, 32),
              cpu::PppmMeshArm::BrickSubgroup);
    EXPECT_EQ(cpu::pppm_select_mesh_arm(31, 8, 1),
              cpu::PppmMeshArm::BrickSubgroup);
}

TEST(PppmSelectMeshArm, ThrowsWhenFftRanksExceedNprocs) {
    EXPECT_THROW(cpu::pppm_select_mesh_arm(32, 16, 32), std::invalid_argument);
    EXPECT_THROW(cpu::pppm_select_mesh_arm(32, 4, 8), std::invalid_argument);
}
