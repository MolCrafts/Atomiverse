/// @file tests/cpu/longrange/test_pme_coul.cpp
/// @brief Validation of CPU Particle-Mesh Ewald (CoulLong + KSpacePME).
///
/// Same checks as the direct-Ewald test, plus a direct PME-vs-Ewald total
/// energy comparison. Built only with ATV_BUILD_KSPACE (FFTW3).

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover_npt.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
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
    frame.simbox = ATV::cpu::Box::cubic(n * d);
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

// Combined real-space + a reciprocal component (templated on the kspace type).
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

// Reciprocal-only PME: isolates self + recip + net from the real-space half.
// Returns energy; leaves system virial in f.meta.get(kStress).
double pme_reciprocal(cpu::Frame& f, double alpha, double spacing = 0.5,
                      int order = 6) {
    zero_force_energy_stress(f);
    cpu::Context ctx;
    cpu::KSpacePME pme(alpha, spacing, order);
    pme.build({ctx, f});
    pme.launch({ctx, f, 0, 0});
    pme.destroy({ctx, f});
    return f.meta.get(ATV::kEnergy);
}

double ewald_reciprocal(cpu::Frame& f, double alpha, double accuracy = 1e-8) {
    zero_force_energy_stress(f);
    cpu::Context ctx;
    cpu::KSpaceEwald ewald(alpha, accuracy);
    ewald.build({ctx, f});
    ewald.launch({ctx, f, 0, 0});
    ewald.destroy({ctx, f});
    return f.meta.get(ATV::kEnergy);
}

// Affine uniaxial strain along axis a: L_a and r_a scale by (1+eps).
// Fractional coordinates stay fixed → ∂E/∂ε_a = virial_aa.
void apply_uniaxial_strain(cpu::Frame& f, int axis, double eps) {
    const double s = 1.0 + eps;
    auto x = f.span(ATV::kAtom_x);
    auto y = f.span(ATV::kAtom_y);
    auto z = f.span(ATV::kAtom_z);
    for (int i = 0; i < f.nlocal(); ++i) {
        if (axis == 0)
            x[i] *= s;
        else if (axis == 1)
            y[i] *= s;
        else
            z[i] *= s;
    }
    double sx = 1.0, sy = 1.0, sz = 1.0;
    if (axis == 0)
        sx = s;
    else if (axis == 1)
        sy = s;
    else
        sz = s;
    f.simbox.scale_diagonal(sx, sy, sz);
}

} // namespace

TEST(PmeCoul, MadelungNaCl) {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 1.9 * d;
    const double alpha =
        3.5 / cutoff; // moderate α so the grid resolves k-space
    const double e = electrostatic_energy(frame, pairs, alpha, cutoff,
                                          cpu::KSpacePME(alpha, 0.5, 6));

    const int N = n * n * n;
    const double e_expected =
        -0.5 * N * kMadelungNaCl * ATV::units::coulomb / d;
    EXPECT_NEAR(e / e_expected, 1.0, 5e-4)
        << "E=" << e << " expected=" << e_expected;
}

TEST(PmeCoul, AgreesWithEwald) {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 1.9 * d;
    const double alpha = 3.5 / cutoff;
    const double e_ewald = electrostatic_energy(frame, pairs, alpha, cutoff,
                                                cpu::KSpaceEwald(alpha, 1e-8));
    const double e_pme = electrostatic_energy(frame, pairs, alpha, cutoff,
                                              cpu::KSpacePME(alpha, 0.5, 6));
    EXPECT_NEAR(e_pme / e_ewald, 1.0, 5e-4)
        << "ewald=" << e_ewald << " pme=" << e_pme;
}

TEST(PmeCoul, FiniteDifferenceForces) {
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
    frame.simbox = ATV::cpu::Box::cubic(8.0);
    cpu::md::PairList pairs;
    build_full_halflist(frame, pairs);

    const double cutoff = 3.9;
    const double alpha = 0.45;
    auto run = [&]() {
        return electrostatic_energy(frame, pairs, alpha, cutoff,
                                    cpu::KSpacePME(alpha, 0.7, 6));
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
            EXPECT_NEAR(f_analytic, -grad, 2e-3)
                << "atom " << a << " axis " << axis;
        }
}

TEST(PmeLifecycle, SeparatesBuildCellRefreshAndCapacityRebuild) {
    cpu::Frame frame(0);
    frame.simbox = cpu::Box::cubic(8.0);
    cpu::Context ctx;
    cpu::BuildContext build_ctx{ctx, frame};
    cpu::LaunchContext launch_ctx{ctx, frame, 0, 0};
    cpu::KSpacePME pme(0.4, 1.0, 6);

    pme.build(build_ctx);
    EXPECT_EQ(pme.lifecycle_stats().builds, 1u);
    EXPECT_EQ(pme.lifecycle_stats().cell_refreshes, 0u);
    EXPECT_EQ(pme.lifecycle_stats().capacity_rebuilds, 0u);
    EXPECT_GT(pme.memory_bytes(), 0u);

    // An unchanged Box revision is a no-op for the lifecycle state machine.
    pme.launch(launch_ctx);
    EXPECT_EQ(pme.lifecycle_stats().cell_refreshes, 0u);
    EXPECT_EQ(pme.lifecycle_stats().capacity_rebuilds, 0u);

    // 8.0 -> 7.92 Angstrom retains the same 8^3 FFT shape.
    frame.simbox.scale_isotropic(0.99);
    pme.launch(launch_ctx);
    EXPECT_EQ(pme.lifecycle_stats().cell_refreshes, 1u);
    EXPECT_EQ(pme.lifecycle_stats().capacity_rebuilds, 0u);

    // A larger cell crosses the mesh-capacity boundary and replans once.
    frame.simbox.scale_isotropic(2.0);
    pme.launch(launch_ctx);
    EXPECT_EQ(pme.lifecycle_stats().cell_refreshes, 1u);
    EXPECT_EQ(pme.lifecycle_stats().capacity_rebuilds, 1u);

    pme.destroy({ctx, frame});
    EXPECT_EQ(pme.memory_bytes(), 0u);
}

TEST(PmeLifecycle, IsReciprocalLongRangeParticipant) {
    cpu::KSpacePME pme(0.4, 1.0, 6);
    const auto* lr = pme.as_long_range_participant();
    ASSERT_NE(lr, nullptr);
    EXPECT_EQ(lr->long_range_contract().role,
              cpu::LongRangeRole::ReciprocalSpace);
}

// ac-008: after warmup, same-shape cell refresh is replan-free and does not
// grow KSpace-owned storage. Capacity rebuild is a separate, counted event.
TEST(PmeLifecycle, StableGridRefreshIsReplanFreeAndMemoryStable) {
    cpu::Frame frame(8);
    // Nonzero charges so the hot path is real (not a zero-charge no-op).
    for (int i = 0; i < 8; ++i) {
        frame.span(ATV::kAtom_x)[i] = 0.5 * (i % 2);
        frame.span(ATV::kAtom_y)[i] = 0.5 * ((i / 2) % 2);
        frame.span(ATV::kAtom_z)[i] = 0.5 * (i / 4);
        frame.span(ATV::kAtom_charge)[i] = (i % 2 == 0) ? 1.0 : -1.0;
        frame.span(ATV::kAtom_mass)[i] = 1.0;
        frame.span(ATV::kAtom_type)[i] = 0;
        frame.span(ATV::kAtom_tag)[i] = i + 1;
    }
    frame.simbox = cpu::Box::cubic(10.0);
    cpu::Context ctx;
    cpu::KSpacePME pme(/*alpha=*/0.4, /*grid_spacing=*/1.0, /*order=*/6);
    pme.build({ctx, frame});
    // Warmup launches (steady storage).
    for (int i = 0; i < 3; ++i)
        pme.launch({ctx, frame, 0, 0});

    const auto after_warm = pme.lifecycle_stats();
    const auto bytes_warm = pme.memory_bytes();
    EXPECT_EQ(after_warm.builds, 1u);
    EXPECT_EQ(after_warm.capacity_rebuilds, 0u);
    EXPECT_GT(bytes_warm, 0u);

    // Same FFT shape: 10 → 9.9 Å keeps mesh; only cell_refresh.
    frame.simbox.scale_isotropic(0.99);
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(ATV::kAtom_x)[i] *= 0.99;
        frame.span(ATV::kAtom_y)[i] *= 0.99;
        frame.span(ATV::kAtom_z)[i] *= 0.99;
    }
    for (int i = 0; i < 5; ++i)
        pme.launch({ctx, frame, 0, 0});

    const auto after_refresh = pme.lifecycle_stats();
    EXPECT_EQ(after_refresh.builds, 1u);
    EXPECT_EQ(after_refresh.capacity_rebuilds, 0u);
    EXPECT_GE(after_refresh.cell_refreshes, 1u);
    EXPECT_EQ(pme.memory_bytes(), bytes_warm)
        << "stable-grid refresh must not grow KSpace storage";

    // Intentional mesh crossing: capacity_rebuild increments; plans recreated.
    frame.simbox.scale_isotropic(2.0);
    pme.launch({ctx, frame, 0, 0});
    EXPECT_EQ(pme.lifecycle_stats().capacity_rebuilds, 1u);
    EXPECT_GE(pme.memory_bytes(), bytes_warm);

    pme.destroy({ctx, frame});
    EXPECT_EQ(pme.memory_bytes(), 0u);
}

TEST(PmeLifecycle, FftPlanOwnershipIsMoveOnly) {
    static_assert(!std::is_copy_constructible_v<cpu::KSpacePME>);
    static_assert(!std::is_copy_assignable_v<cpu::KSpacePME>);
    static_assert(std::is_nothrow_move_constructible_v<cpu::KSpacePME>);
}

TEST(PmeLifecycle, BuiltFftPlansTransferToExactlyOneOwner) {
    cpu::Frame frame(0);
    frame.simbox = cpu::Box::cubic(8.0);
    cpu::Context ctx;
    cpu::KSpacePME source(0.4, 1.0, 6);
    source.build({ctx, frame});
    const auto bytes = source.memory_bytes();

    cpu::KSpacePME destination(std::move(source));
    EXPECT_EQ(destination.memory_bytes(), bytes);
    destination.launch({ctx, frame, 0, 0});
    destination.destroy({ctx, frame});
    EXPECT_EQ(destination.memory_bytes(), 0u);
}

TEST(PmeLifecycle, BoxIdentitySeparatesEqualRevisionReplicaFrames) {
    cpu::Frame first(0);
    cpu::Frame second(0);
    first.simbox = cpu::Box::cubic(8.0);
    second.simbox = cpu::Box::cubic(8.0);
    ASSERT_EQ(first.simbox.revision(), second.simbox.revision());
    ASSERT_NE(&first.simbox, &second.simbox);

    cpu::Context ctx;
    cpu::KSpacePME pme(0.4, 1.0, 6);
    pme.build({ctx, first});
    pme.launch({ctx, second, 0, 0});
    EXPECT_EQ(pme.lifecycle_stats().cell_refreshes, 1u);
    EXPECT_EQ(pme.lifecycle_stats().capacity_rebuilds, 0u);
    pme.destroy({ctx, second});
}

TEST(LongRangePlanContract, ActualEwaldAndPmePairsBuild) {
    constexpr double alpha = 0.3;
    constexpr double cutoff = 8.0;

    cpu::md::MD ewald;
    ewald.set_integrator(cpu::md::VelocityVerlet{1.0});
    ewald.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    ewald.on(md::Phase::force).add(cpu::CoulLong{alpha, cutoff});
    ewald.add(cpu::KSpaceEwald{alpha, 1e-5});
    EXPECT_NO_THROW(ewald.build());

    cpu::md::MD pme;
    pme.set_integrator(cpu::md::VelocityVerlet{1.0});
    pme.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    pme.on(md::Phase::force).add(cpu::CoulLong{alpha, cutoff});
    pme.add(cpu::KSpacePME{alpha, 1.0, 6});
    EXPECT_NO_THROW(pme.build());
}

TEST(LongRangePlanContract, ActualMismatchedAlphaFailsBeforePlanBuild) {
    constexpr double cutoff = 8.0;
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{1.0});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    sim.on(md::Phase::force).add(cpu::CoulLong{0.3, cutoff});
    sim.add(cpu::KSpacePME{0.31, 1.0, 6});
    EXPECT_THROW(sim.build(), std::runtime_error);
}

TEST(LongRangePlanContract, NptAcceptsPmeWithReciprocalStress) {
    constexpr double alpha = 0.3;
    constexpr double cutoff = 8.0;
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::NoseHooverNPT{{.T_target = 300.0,
                                               .tau_T = 100.0,
                                               .P_target = 0.0,
                                               .tau_P = 1000.0,
                                               .dt = 1.0}});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    sim.on(md::Phase::force).add(cpu::CoulLong{alpha, cutoff});
    sim.add(cpu::KSpacePME{alpha, 1.0, 6});
    EXPECT_NO_THROW(sim.build());
}

// ── Reciprocal virial: PME vs Ewald and finite-strain oracle ───────────

TEST(PmeStress, AgreesWithEwaldOnRockSalt) {
    const double d = 2.82;
    const int n = 4;
    auto frame_pme = make_rocksalt(d, n);
    auto frame_ew = make_rocksalt(d, n);
    const double cutoff = 1.9 * d;
    const double alpha = 3.5 / cutoff;
    cpu::md::PairList pairs;
    build_full_halflist(frame_pme, pairs);

    // Full electrostatics (real + reciprocal) so the comparison is what NPT
    // would actually see: pair virial + reciprocal virial.
    electrostatic_energy(frame_pme, pairs, alpha, cutoff,
                         cpu::KSpacePME(alpha, 0.5, 6));
    electrostatic_energy(frame_ew, pairs, alpha, cutoff,
                         cpu::KSpaceEwald(alpha, 1e-8));

    for (int s = 0; s < 6; ++s) {
        const double p = frame_pme.meta.get(ATV::kStress)[s];
        const double e = frame_ew.meta.get(ATV::kStress)[s];
        // Rock-salt: diagonal ~ equal magnitude, off-diagonal ~ 0. Grid
        // discretization dominates; 1% relative on the large diagonal, and
        // an absolute floor for the near-zero shear components.
        const double scale = std::max(1.0, std::fabs(e));
        EXPECT_NEAR(p, e, 5e-3 * scale)
            << "Voigt component " << s << " pme=" << p << " ewald=" << e;
    }
}

TEST(PmeStress, FiniteStrainDiagonalMatchesVirial) {
    // Reciprocal-only: self is volume-independent and the cell is neutral.
    // Affine uniaxial strain at fixed fractional coordinates gives the
    // force-virial convention used by CoulLong / barostats:
    //   V_αα = −∂E/∂ε_α
    // (cohesive Madelung: expand → energy rises, ∂E/∂ε > 0, virial < 0).
    const double d = 2.82;
    auto frame0 = make_rocksalt(d, /*n=*/4);
    const double alpha = 3.5 / (1.9 * d);

    pme_reciprocal(frame0, alpha, 0.5, 6);
    double vir[3] = {frame0.meta.get(ATV::kStress)[0],
                     frame0.meta.get(ATV::kStress)[1],
                     frame0.meta.get(ATV::kStress)[2]};

    // Off-diagonal reciprocal virial vanishes by cubic symmetry.
    for (int s = 3; s < 6; ++s)
        EXPECT_NEAR(frame0.meta.get(ATV::kStress)[s], 0.0, 1e-4)
            << "Voigt shear " << s;

    const double h = 1e-5;
    for (int axis = 0; axis < 3; ++axis) {
        auto fp = make_rocksalt(d, 4);
        auto fm = make_rocksalt(d, 4);
        apply_uniaxial_strain(fp, axis, +h);
        apply_uniaxial_strain(fm, axis, -h);
        const double ep = pme_reciprocal(fp, alpha, 0.5, 6);
        const double em = pme_reciprocal(fm, alpha, 0.5, 6);
        const double dE_deps = (ep - em) / (2.0 * h);
        const double fd_virial = -dE_deps;
        // Grid+spline discretization + finite-h truncation; 1% of |virial|.
        const double tol = 1e-2 * std::max(1.0, std::fabs(vir[axis]));
        EXPECT_NEAR(vir[axis], fd_virial, tol)
            << "axis " << axis << " virial=" << vir[axis]
            << " -dE/dε=" << fd_virial;
    }
}

TEST(PmeStress, ScaledCellMatchesFreshBuild) {
    // ac-003: cell refresh must recompute energy/force/stress to match a
    // freshly built component at the new cell (no stale influence data).
    const double d = 2.82;
    auto frame = make_rocksalt(d, /*n=*/4);
    const double alpha = 3.5 / (1.9 * d);
    cpu::Context ctx;
    cpu::KSpacePME pme(alpha, 0.5, 6);
    pme.build({ctx, frame});

    auto eval = [&](cpu::Frame& f) {
        zero_force_energy_stress(f);
        pme.launch({ctx, f, 0, 0});
        return f.meta.get(ATV::kEnergy);
    };

    // Reference at original cell via the persistent component.
    const double e0 = eval(frame);
    double s0[6];
    for (int s = 0; s < 6; ++s)
        s0[s] = frame.meta.get(ATV::kStress)[s];

    // Uniform scale that keeps the FFT shape (same grid dimensions).
    frame.simbox.scale_isotropic(0.99);
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(ATV::kAtom_x)[i] *= 0.99;
        frame.span(ATV::kAtom_y)[i] *= 0.99;
        frame.span(ATV::kAtom_z)[i] *= 0.99;
    }
    const double e_reused = eval(frame);
    double s_reused[6];
    for (int s = 0; s < 6; ++s)
        s_reused[s] = frame.meta.get(ATV::kStress)[s];

    // Fresh component at the scaled cell.
    auto fresh = make_rocksalt(d, 4);
    fresh.simbox.scale_isotropic(0.99);
    for (int i = 0; i < fresh.nlocal(); ++i) {
        fresh.span(ATV::kAtom_x)[i] *= 0.99;
        fresh.span(ATV::kAtom_y)[i] *= 0.99;
        fresh.span(ATV::kAtom_z)[i] *= 0.99;
    }
    const double e_fresh = pme_reciprocal(fresh, alpha, 0.5, 6);

    EXPECT_NE(e_reused, e0); // cell change must move the energy
    EXPECT_NEAR(e_reused, e_fresh, 1e-10 * std::max(1.0, std::fabs(e_fresh)));
    for (int s = 0; s < 6; ++s) {
        EXPECT_NEAR(s_reused[s], fresh.meta.get(ATV::kStress)[s],
                    1e-10 * std::max(1.0, std::fabs(s_reused[s])))
            << "Voigt " << s;
        // And stress itself must have changed under the scale.
        if (s < 3)
            EXPECT_NE(s_reused[s], s0[s]);
    }
    EXPECT_EQ(pme.lifecycle_stats().cell_refreshes, 1u);
    pme.destroy({ctx, frame});
}
