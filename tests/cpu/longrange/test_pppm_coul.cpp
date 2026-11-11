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
///   7. Energy and forces equal a literal LAMMPS make_rho / fieldforce_ik
///      reference (tests/core/pppm_lammps_reference.h) for odd and even
///      orders: the centred charge-assignment stencil

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/core/longrange_math.h"
#include "atomiverse/core/pppm_ik.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

#include "core/pppm_lammps_reference.h"
#include "periodic_rocksalt.h"

using namespace ATV;
using ATV::test::build_dense_halflist;
using ATV::test::electrostatic_energy;
using ATV::test::kMadelungNaCl;
using ATV::test::make_rocksalt;
using ATV::test::PeriodicRockSalt;
using ATV::test::zero_force_energy_stress;

namespace {} // namespace

TEST(PppmCoul, MadelungNaCl) {
    const double d = 2.82;
    const int n = 4;
    const double cutoff = 1.9 * d;
    PeriodicRockSalt rs(d, n, cutoff);

    // Moderate α + fine mesh so HE G(k) resolves the lattice well.
    const double alpha = 3.5 / cutoff;
    const double e = electrostatic_energy(
        rs.frame, rs.context, alpha, cutoff,
        cpu::KSpacePPPM(alpha, /*spacing=*/0.5, /*order=*/5));

    const int N = n * n * n;
    const double e_expected =
        -0.5 * N * kMadelungNaCl * ATV::units::UnitSystem::lj().qqrd2e() / d;
    EXPECT_NEAR(e / e_expected, 1.0, 1e-3)
        << "E=" << e << " expected=" << e_expected;
}

TEST(PppmCoul, AgreesWithEwald) {
    const double d = 2.82;
    const int n = 4;
    const double cutoff = 1.9 * d;
    PeriodicRockSalt rs(d, n, cutoff);

    const double alpha = 3.5 / cutoff;
    const double e_ewald = electrostatic_energy(
        rs.frame, rs.context, alpha, cutoff, cpu::KSpaceEwald(alpha, 1e-8));
    const double e_pppm = electrostatic_energy(
        rs.frame, rs.context, alpha, cutoff, cpu::KSpacePPPM(alpha, 0.5, 5));
    EXPECT_NEAR(e_pppm / e_ewald, 1.0, 1e-3)
        << "ewald=" << e_ewald << " pppm=" << e_pppm;
}

TEST(PppmCoul, AgreesWithPme) {
    const double d = 2.82;
    const int n = 4;
    const double cutoff = 1.9 * d;
    PeriodicRockSalt rs(d, n, cutoff);

    const double alpha = 3.5 / cutoff;
    const double e_pme = electrostatic_energy(
        rs.frame, rs.context, alpha, cutoff, cpu::KSpacePME(alpha, 0.5, 6));
    const double e_pppm = electrostatic_energy(
        rs.frame, rs.context, alpha, cutoff, cpu::KSpacePPPM(alpha, 0.5, 5));
    EXPECT_NEAR(e_pppm / e_pme, 1.0, 2e-3)
        << "pme=" << e_pme << " pppm=" << e_pppm;
}

TEST(PppmCoul, AlphaIndependence) {
    const double d = 2.82;
    const int n = 4;
    const double cutoff = 1.9 * d;
    PeriodicRockSalt rs(d, n, cutoff);

    // Fine mesh so both α values are well-resolved.
    const double e1 =
        electrostatic_energy(rs.frame, rs.context, 3.0 / cutoff, cutoff,
                             cpu::KSpacePPPM(3.0 / cutoff, 0.4, 5));
    const double e2 =
        electrostatic_energy(rs.frame, rs.context, 5.0 / cutoff, cutoff,
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
    // Every pair is nearer than any periodic image, so the dense owner-only
    // list is complete.
    frame.box = ATV::cpu::Box::cubic(8.0);
    cpu::md::PairList pairs;
    build_dense_halflist(frame, pairs);
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;

    const double cutoff = 3.9;
    const double alpha = 0.45;
    auto run = [&]() {
        return electrostatic_energy(frame, ctx, alpha, cutoff,
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

namespace {

// Ten charges at fixed fractional coordinates; ±0.8 e, neutral.
constexpr int kStencilAtoms = 10;
std::vector<double> stencil_frac() {
    std::vector<double> s(3 * kStencilAtoms);
    for (int i = 0; i < kStencilAtoms; ++i) {
        s[3 * i + 0] = std::fmod(0.137 + 0.618034 * i, 1.0);
        s[3 * i + 1] = std::fmod(0.291 + 0.414214 * i, 1.0);
        s[3 * i + 2] = std::fmod(0.533 + 0.732051 * i, 1.0);
    }
    return s;
}
std::vector<double> stencil_charge() {
    std::vector<double> q(kStencilAtoms);
    for (int i = 0; i < kStencilAtoms; ++i)
        q[i] = i % 2 ? -0.8 : 0.8;
    return q;
}

// cpu::KSpacePPPM against the LAMMPS reference (full complex mesh, real part
// of each ik transform): energy, forces and the six-component virial. The
// odd 9×9×15 mesh has no Nyquist planes; the even 8×8×12 one has all three,
// at a spacing coarse enough (G at the Nyquist index ≈ 1e-2 of the peak) for
// a wrong Nyquist fold to show at the 1e-3 level.
void check_stencil_against_lammps(const double* cell, int order, double spacing,
                                  const int (&K_expect)[3]) {
    SCOPED_TRACE("order " + std::to_string(order) + ", mesh "
                 + std::to_string(K_expect[0]) + "x"
                 + std::to_string(K_expect[1]) + "x"
                 + std::to_string(K_expect[2]));
    constexpr double kAlpha = 0.9;
    const double kSpacing = spacing;
    const std::vector<double> frac = stencil_frac();
    const std::vector<double> q = stencil_charge();
    cpu::Frame f(kStencilAtoms);
    Mat3 h;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            h(r, c) = cell[3 * r + c];
    f.box = cpu::Box(h);
    const ColKey<StorageReal>* axis[3] = {&kAtom_x, &kAtom_y, &kAtom_z};
    double qsq = 0.0;
    for (int i = 0; i < kStencilAtoms; ++i) {
        for (int r = 0; r < 3; ++r) {
            double x = 0.0;
            for (int c = 0; c < 3; ++c)
                x += cell[3 * r + c] * frac[3 * i + c];
            f.span(*axis[r])[i] = x;
        }
        f.span(kAtom_charge)[i] = q[i];
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
        qsq += q[i] * q[i];
    }
    int K[3];
    for (int d = 0; d < 3; ++d)
        K[d] = longrange::pppm_fft_size(
            static_cast<int>(std::ceil(f.box.lengths()[d] / kSpacing)));
    for (int d = 0; d < 3; ++d)
        ASSERT_EQ(K[d], K_expect[d]);

    cpu::KSpacePPPM solver(kAlpha, kSpacing, order);
    cpu::Context runtime;
    solver.build({runtime, f});
    zero_force_energy_stress(f);
    solver.launch({runtime, f, 0, 0});
    const double e_atv = f.meta.get(kEnergy)
                         - longrange::self_energy_prefactor(kAlpha, 1.0) * qsq;
    double stress_atv[6];
    for (int c = 0; c < 6; ++c)
        stress_atv[c] = f.meta.get(kStress)[c];
    solver.destroy({runtime, f});

    namespace ref = ATV::test::lammps_pppm;
    const ref::Result lammps = ref::solve(cell, frac, q, K, kAlpha, order);
    double fmax = 0.0, dfmax = 0.0;
    for (int i = 0; i < kStencilAtoms; ++i)
        for (int r = 0; r < 3; ++r) {
            const AccReal* fr[3] = {f.span(kAtom_fx), f.span(kAtom_fy),
                                    f.span(kAtom_fz)};
            fmax = std::max(fmax, std::abs(lammps.force[3 * i + r]));
            dfmax =
                std::max(dfmax, std::abs(fr[r][i] - lammps.force[3 * i + r]));
        }
    EXPECT_NEAR(e_atv, lammps.energy, 1e-12 * std::abs(lammps.energy));
    EXPECT_LT(dfmax, 1e-11 * fmax);
    double wmax = 0.0, dwmax = 0.0;
    for (int c = 0; c < 6; ++c) {
        wmax = std::max(wmax, std::abs(lammps.virial[c]));
        dwmax = std::max(dwmax, std::abs(stress_atv[c] - lammps.virial[c]));
    }
    EXPECT_GT(wmax, 0.0);
    EXPECT_LT(dwmax, 1e-12 * wmax);

    // The reference separates the stencils: the one-sided smooth-PME
    // placement (u − order/2) is the same discretization for even orders and
    // a different one for odd orders.
    const ref::Result one_sided =
        ref::solve(cell, frac, q, K, kAlpha, order, 0.5 * order);
    const double de =
        std::abs(one_sided.energy - lammps.energy) / std::abs(lammps.energy);
    if (order % 2)
        EXPECT_GT(de, 1e-6);
    else
        EXPECT_LT(de, 1e-12);
}

} // namespace

TEST(PppmStencil, OrthorhombicMatchesLammpsMakeRhoAndFieldforce) {
    const double cell[9] = {6.3, 0, 0, 0, 6.3, 0, 0, 0, 10.4};
    for (int order : {3, 4, 5, 6, 7})
        check_stencil_against_lammps(cell, order, 0.7, {9, 9, 15});
}

TEST(PppmStencil, TriclinicMatchesLammpsMakeRhoAndFieldforce) {
    const double cell[9] = {6.3, 0.7, -0.6, 0, 6.25, 0.9, 0, 0, 10.4};
    for (int order : {4, 5, 7})
        check_stencil_against_lammps(cell, order, 0.7, {9, 9, 15});
}

// Even meshes: LAMMPS's ik derivative along a Nyquist index cancels in the
// real part of its complex transform; the r2c fold must reproduce that, and
// the energy and virial of the conjugate pair.
TEST(PppmStencil, EvenMeshNyquistPlanesMatchLammpsPoissonIk) {
    const double cell[9] = {6.4, 0, 0, 0, 6.4, 0, 0, 0, 9.6};
    for (int order : {3, 4, 5, 6, 7})
        check_stencil_against_lammps(cell, order, 0.8, {8, 8, 12});
}

TEST(PppmStencil, EvenMeshTriclinicNyquistPlanesMatchLammpsPoissonIk) {
    const double cell[9] = {6.4, 0.7, -0.6, 0, 6.35, 0.9, 0, 0, 9.5};
    for (int order : {4, 5, 6})
        check_stencil_against_lammps(cell, order, 0.8, {8, 8, 12});
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

TEST(KSpaceRequests, ForcesAndStressAreIndependent) {
    auto check = [](auto solver) {
        SCOPED_TRACE(solver.name());
        auto f = make_rocksalt(2.82, 2);
        f.span(kAtom_x)[0] += 0.17;
        cpu::Context runtime;
        solver.build({runtime, f});
        zero_force_energy_stress(f);
        solver.launch({runtime, f, 0, 0});
        const auto baseline = f;
        ASSERT_GT(std::abs(f.span(kAtom_fx)[0]), 1e-5);
        for (bool forces : {false, true})
            for (bool stress : {false, true}) {
                zero_force_energy_stress(f);
                for (int i = 0; i < f.nlocal(); ++i)
                    f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] =
                        f.span(kAtom_fz)[i] = 7;
                f.meta.get(kStress).fill(11);
                solver.launch({runtime, f, 0, 0, forces, stress});
                EXPECT_NEAR(f.meta.get(kEnergy), baseline.meta.get(kEnergy),
                            1e-12);
                for (int i = 0; i < f.nlocal(); ++i) {
                    EXPECT_NEAR(f.span(kAtom_fx)[i],
                                7 + (forces ? baseline.span(kAtom_fx)[i] : 0),
                                1e-12);
                    EXPECT_NEAR(f.span(kAtom_fy)[i],
                                7 + (forces ? baseline.span(kAtom_fy)[i] : 0),
                                1e-12);
                    EXPECT_NEAR(f.span(kAtom_fz)[i],
                                7 + (forces ? baseline.span(kAtom_fz)[i] : 0),
                                1e-12);
                }
                for (int d = 0; d < 6; ++d)
                    EXPECT_NEAR(
                        f.meta.get(kStress)[d],
                        11 + (stress ? baseline.meta.get(kStress)[d] : 0),
                        1e-12);
            }
        solver.destroy({runtime, f});
    };
    check(cpu::KSpaceEwald{0.5, 1e-9});
    check(cpu::KSpacePME{0.5, 0.7, 6});
    check(cpu::KSpacePPPM{0.5, 0.7, 6});
}

TEST(KSpaceVirial, TraceIsNegativeAffineEnergyDerivativeIncludingNetCharge) {
    auto check = [](auto make_solver, bool charged) {
        auto evaluate = [&](double scale, bool stress) {
            auto f = make_rocksalt(2.82 * scale, 2);
            f.span(kAtom_x)[0] += 0.17 * scale;
            if (charged)
                f.span(kAtom_charge)[0] += 0.4;
            // Scale spacing together with the box to keep the FFT mesh fixed.
            auto solver = make_solver(scale);
            cpu::Context runtime;
            solver.build({runtime, f});
            zero_force_energy_stress(f);
            solver.launch({runtime, f, 0, 0, false, true});
            const double result = stress ? f.meta.get(kStress)[0]
                                               + f.meta.get(kStress)[1]
                                               + f.meta.get(kStress)[2]
                                         : f.meta.get(kEnergy);
            solver.destroy({runtime, f});
            return result;
        };
        const double h = 1e-5;
        const double derivative =
            -(evaluate(1 + h, false) - evaluate(1 - h, false)) / (2 * h);
        EXPECT_NEAR(evaluate(1, true), derivative, 2e-7);
    };
    for (bool charged : {false, true}) {
        check([](double) { return cpu::KSpaceEwald{0.5, 1e-10}; }, charged);
        check([](double s) { return cpu::KSpacePME{0.5, 0.7 * s, 6}; },
              charged);
        check([](double s) { return cpu::KSpacePPPM{0.5, 0.7 * s, 6}; },
              charged);
    }
}

#if defined(ATV_HAS_MPI) && defined(ATV_HAS_FFTW_MPI)
TEST(PppmCoul, NonuniformDistributedCutsRejectBeforeBuildOrCollectives) {
    // Synthetic published Runtime metadata: rejection must not call MPI/FFT.
    // Restore the caller's explicit mesh choice, including on assertion failure.
    struct EnvRestore {
        bool present = std::getenv("ATV_PPPM_ALLREDUCE") != nullptr;
        std::string value = present ? std::getenv("ATV_PPPM_ALLREDUCE") : "";
        EnvRestore() { unsetenv("ATV_PPPM_ALLREDUCE"); }
        ~EnvRestore() {
            if (present)
                setenv("ATV_PPPM_ALLREDUCE", value.c_str(), 1);
            else
                unsetenv("ATV_PPPM_ALLREDUCE");
        }
    } restore;
    auto frame = make_rocksalt(2.8, 2);
    cpu::md::mpi::DomainDecomposition dd;
    dd.dims[0] = 2;
    dd.dims[1] = dd.dims[2] = 1;
    dd.cart_comm = MPI_COMM_WORLD;
    dd.cuts[0] = {0.0, 0.3, 1.0};
    cpu::Context ctx;
    ctx.mpi_size = 2;
    ctx.md.dd = &dd;
    cpu::KSpacePPPM mesh(0.3, 0.8, 4);
    frame.meta.get(kEnergy) = 17.0;
    EXPECT_THROW(mesh.build({ctx, frame}), std::invalid_argument);
    EXPECT_THROW(mesh.launch({ctx, frame, 0, 0}), std::invalid_argument);
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 17.0);
}
#endif

#ifdef _OPENMP
namespace {

struct KSpaceSnapshot {
    double energy = 0.0;
    double stress[6] = {};
    std::vector<double> f;

    bool operator==(const KSpaceSnapshot& o) const {
        for (int k = 0; k < 6; ++k)
            if (stress[k] != o.stress[k])
                return false;
        return energy == o.energy && f == o.f;
    }
};

/// Repeated launches of one built reciprocal solver at OMP 1 / 4 / 8 must
/// agree bit for bit. (The energy / virial fold is thread-count independent
/// by construction; the FFT and so the forces follow FFTW's threaded plan.)
template <class Make>
void expect_kspace_reproducible(Make make) {
    const int saved = omp_get_max_threads();
    const int saved_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    for (const int threads : {1, 4, 8}) {
        omp_set_num_threads(threads);
        cpu::Frame f = make_rocksalt(2.82, 8);
        // Break the lattice symmetry so no sum is trivially exact.
        for (int i = 0; i < f.nlocal(); ++i) {
            f.span(kAtom_x)[i] += 0.05 * std::sin(1.3 * i);
            f.span(kAtom_y)[i] += 0.05 * std::cos(0.7 * i);
            f.span(kAtom_z)[i] += 0.05 * std::sin(2.1 * i + 0.4);
        }
        cpu::Context ctx;
        auto recip = make();
        recip.build({ctx, f});
        std::vector<KSpaceSnapshot> runs;
        for (int rep = 0; rep < 4; ++rep) {
            zero_force_energy_stress(f);
            recip.launch({ctx, f, 0, 0, true, true});
            KSpaceSnapshot s;
            s.energy = f.meta.get(kEnergy);
            for (int k = 0; k < 6; ++k)
                s.stress[k] = f.meta.get(kStress)[k];
            for (int i = 0; i < f.nlocal(); ++i) {
                s.f.push_back(f.span(kAtom_fx)[i]);
                s.f.push_back(f.span(kAtom_fy)[i]);
                s.f.push_back(f.span(kAtom_fz)[i]);
            }
            runs.push_back(std::move(s));
        }
        recip.destroy({ctx, f});
        EXPECT_NE(runs[0].energy, 0.0);
        for (int rep = 1; rep < 4; ++rep)
            EXPECT_TRUE(runs[rep] == runs[0])
                << threads << " threads, repeat " << rep;
    }
    omp_set_num_threads(saved);
    omp_set_dynamic(saved_dynamic);
}

} // namespace

TEST(PppmCoul, ThreadedRunIsBitReproduciblePerThreadCount) {
    // 512 ions, 48³ mesh: the threaded spread, Green's sum and gather all run
    // with a full team; the energy / virial sums are folded in a fixed chunk
    // order.
    expect_kspace_reproducible(
        [] { return cpu::KSpacePPPM(0.35, /*spacing=*/0.47, /*order=*/5); });
}

TEST(PmeCoul, ThreadedRunIsBitReproduciblePerThreadCount) {
    expect_kspace_reproducible(
        [] { return cpu::KSpacePME(0.35, /*spacing=*/0.47, /*order=*/5); });
}
#endif
