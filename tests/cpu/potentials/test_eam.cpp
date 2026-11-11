/// @file tests/cpu/potentials/test_eam.cpp
/// @brief Unit: analytic EAM (Sutton-Chen) — lookup, density, and the FD force
///        oracle.
///
/// The FD oracle is the hard gate. Energy-only tests cannot see a force that is
/// the gradient of a DIFFERENT functional than the one the energy evaluates —
/// that exact bug hid in the GPW path for 17 commits. Every force this engine
/// grows gets differenced against its own energy.

#include <cmath>
#include <limits>
#include <vector>

#include "../dynamics/periodic_neighbor_fixture.h"
#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/eam.h"
#include "atomiverse/cpu/potentials/eam_potential.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

/// Distorted fcc copper — several call sites below.
cpu::Frame copper(double a0, int n_cell, double jitter) {
    const int n = 4 * n_cell * n_cell * n_cell;
    cpu::Frame f(n);
    const double b[4][3] = {
        {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    int idx = 0;
    for (int ix = 0; ix < n_cell; ++ix)
        for (int iy = 0; iy < n_cell; ++iy)
            for (int iz = 0; iz < n_cell; ++iz)
                for (int k = 0; k < 4; ++k) {
                    const double j = jitter * std::sin(double(13 * idx + 3));
                    f.span(ATV::kAtom_x)[idx] = (double(ix) + b[k][0]) * a0 + j;
                    f.span(ATV::kAtom_y)[idx] =
                        (double(iy) + b[k][1]) * a0 - 0.7 * j;
                    f.span(ATV::kAtom_z)[idx] =
                        (double(iz) + b[k][2]) * a0 + 0.3 * j;
                    f.span(ATV::kAtom_mass)[idx] = 63.546;
                    f.span(ATV::kAtom_type)[idx] = 29;
                    ++idx;
                }
    f.box = ATV::cpu::Box::cubic(a0 * n_cell);
    return f;
}

/// Single-point energy through the public Evaluate driver — called once per FD
/// displacement, i.e. many times.
double energy_of(cpu::Frame& f, double cutoff) {
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
    eval.on(evaluate::Phase::evaluate)
        .add(cpu::EAM{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"},
                      cutoff});
    eval.build();
    eval.run(f);
    return f.meta.get(ATV::kEnergy);
}

} // namespace

/// The lookup constructor throws rather than inventing a fallback.
TEST(EAM, LookupThrowsOnUnsupported) {
    EXPECT_NO_THROW(cpu::EAMPotential(molrs::Element::Cu, "sutton-chen"));
    EXPECT_THROW(
        cpu::EAMPotential(molrs::Element::Cu, "no-such-parameterization"),
        std::out_of_range);
    EXPECT_THROW(cpu::EAMPotential(molrs::Element::C, "sutton-chen"),
                 std::out_of_range);
}

/// The embedding function is nonlinear — the property that makes EAM many-body.
/// If F were linear in rho, EAM would collapse to a pair potential.
TEST(EAM, EmbeddingIsNonlinear) {
    const cpu::EAMPotential p(molrs::Element::Cu, "sutton-chen");
    // F(2 rho) != 2 F(rho) for a sqrt embedding.
    const double rho = 3.0;
    EXPECT_GT(std::fabs(p.F(2.0 * rho) - 2.0 * p.F(rho)), 1e-6);
    // F is negative (binding) and dF is negative and finite away from zero.
    EXPECT_LT(p.F(rho), 0.0);
    EXPECT_LT(p.dF(rho), 0.0);
    // rho = 0 must not produce a NaN even though sqrt has infinite slope there.
    EXPECT_TRUE(std::isfinite(p.dF(0.0)));
}

/// Analytic derivatives of phi and f match finite differences.
TEST(EAM, PairAndDensityDerivativesMatchFD) {
    const cpu::EAMPotential p(molrs::Element::Cu, "sutton-chen");
    const double h = 1e-6;
    for (double r : {2.2, 2.556, 3.0, 4.5, 6.0}) {
        const double dphi_fd = (p.phi(r + h) - p.phi(r - h)) / (2 * h);
        const double df_fd = (p.f(r + h) - p.f(r - h)) / (2 * h);
        EXPECT_NEAR(p.dphi(r), dphi_fd, 1e-6 * std::fabs(dphi_fd) + 1e-9);
        EXPECT_NEAR(p.df(r), df_fd, 1e-6 * std::fabs(df_fd) + 1e-9);
    }
}

/// THE GATE: the analytic many-body force is the gradient of the energy the
/// component actually reports.
TEST(EAM, AnalyticForceMatchesFiniteDifference) {
    const double cutoff = 7.0;
    cpu::Frame f = copper(3.61, 2, /*jitter=*/0.12); // 32 atoms, distorted
    const int n = f.nlocal();

    // Analytic forces.
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
    eval.on(evaluate::Phase::evaluate)
        .add(cpu::EAM{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"},
                      cutoff});
    eval.build();
    eval.run(f);

    std::vector<double> fx(n), fy(n), fz(n);
    for (int i = 0; i < n; ++i) {
        fx[i] = f.span(ATV::kAtom_fx)[i];
        fy[i] = f.span(ATV::kAtom_fy)[i];
        fz[i] = f.span(ATV::kAtom_fz)[i];
    }

    // Central-difference the energy for a handful of atoms (all 3N would be slow
    // and buys nothing — a sign or factor error shows up on the first atom).
    const double h = 1.0e-5;
    const double force_tol = 1.0e-6;
    double worst = 0.0;
    for (int i : {0, 5, 17, 31}) {
        for (int d = 0; d < 3; ++d) {
            auto coord = (d == 0)   ? f.span(ATV::kAtom_x)
                         : (d == 1) ? f.span(ATV::kAtom_y)
                                    : f.span(ATV::kAtom_z);
            const double x0 = coord[i];

            coord[i] = x0 + h;
            cpu::Frame plus = f;
            const double e_plus = energy_of(plus, cutoff);

            coord[i] = x0 - h;
            cpu::Frame minus = f;
            const double e_minus = energy_of(minus, cutoff);

            coord[i] = x0;

            const double f_fd = -(e_plus - e_minus) / (2 * h);
            const double f_an = (d == 0) ? fx[i] : (d == 1) ? fy[i] : fz[i];
            worst = std::max(worst, std::fabs(f_an - f_fd));
            EXPECT_NEAR(f_an, f_fd, force_tol)
                << "atom " << i << " dim " << d << ": analytic " << f_an
                << " vs FD " << f_fd;
        }
    }
    EXPECT_LT(worst, force_tol);
}

/// Newton's third law: the many-body forces still sum to zero.
TEST(EAM, ForcesSumToZero) {
    const double cutoff = 7.0;
    cpu::Frame f = copper(3.61, 2, 0.12);
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3, 1, 0, true});
    eval.on(evaluate::Phase::evaluate)
        .add(cpu::EAM{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"},
                      cutoff});
    eval.build();
    eval.run(f);

    double sx = 0, sy = 0, sz = 0;
    for (int i = 0; i < f.nlocal(); ++i) {
        sx += f.span(ATV::kAtom_fx)[i];
        sy += f.span(ATV::kAtom_fy)[i];
        sz += f.span(ATV::kAtom_fz)[i];
    }
    EXPECT_NEAR(sx, 0.0, 1e-9);
    EXPECT_NEAR(sy, 0.0, 1e-9);
    EXPECT_NEAR(sz, 0.0, 1e-9);
}

TEST(EAM, SharedHalfNeighborListPreservesLJCounting) {
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(20.0);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_x)[i] = 1.0 + 3.0 * i;
        frame.span(kAtom_y)[i] = 1.0;
        frame.span(kAtom_z)[i] = 1.0;
        frame.span(kAtom_mass)[i] = 63.546;
        frame.span(kAtom_type)[i] = 29;
    }
    cpu::md::NeighborList<> neighbors(5.0, 0.3);
    cpu::EAM eam(cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"}, 5.0);
    cpu::PairLJCut lj(1.0, 2.0, 5.0);
    PlanBuilder<cpu::Component> builder;
    builder.add({0, 0, &neighbors});
    builder.add({0, 1, &eam});
    builder.add({0, 2, &lj});
    const auto plan = builder.build(1);
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(plan);
    runtime.set_frame(frame);
    runtime.build_components();
    runtime.context().md.comm->setup({runtime.context(), frame, 0, 0});
    runtime.context().md.comm->borders({runtime.context(), frame, 0, 0});
    neighbors.launch({runtime.context(), frame, 0, 0});
    frame.meta.get(kEnergy) = 0.0;
    lj.launch({runtime.context(), frame, 0, 0});
    const double s6 = std::pow(2.0 / 3.0, 6);
    const double c6 = std::pow(2.0 / 5.0, 6);
    EXPECT_NEAR(frame.meta.get(kEnergy), 4.0 * (s6 * s6 - s6 - c6 * c6 + c6),
                1e-13);
    runtime.destroy_components();
}

TEST(EAM, CutoffMustBeFiniteAndPositive) {
    const cpu::EAMPotential pot{molrs::Element::Cu, "sutton-chen"};
    for (double cutoff : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW((cpu::EAM{pot, cutoff}), std::invalid_argument);
}

TEST(EAM, ResolvedSkinImagesStayOutsideBothEAMAndLJKernels) {
    // Nine partners exercise SIMD batches and the scalar tail on NEON/AVX.
    // Only the first is within rc. Every far row would fold from 4.0 to 3.6
    // if the kernel ignored the source list's already-resolved x images.
    constexpr int count = 10;
    constexpr double r = 3.6, cutoff = 3.77118;
    const cpu::EAMPotential pot{molrs::Element::Cu, "sutton-chen"};
    const double eam_energy =
        pot.phi(r) - pot.phi(cutoff) + 2 * pot.F(pot.f(r));
    const double eam_derivative =
        pot.dphi(r) + 2 * pot.dF(pot.f(r)) * pot.df(r);
    const double s6 = std::pow(2.0 / r, 6), c6 = std::pow(2.0 / cutoff, 6);
    const double lj_energy = 0.4 * (s6 * s6 - s6 - c6 * c6 + c6);
    const double lj_derivative = -2.4 * (2 * s6 * s6 - s6) / r;
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
#endif
    for (int threads : {1, 2}) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        {
            for (bool filtered : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "threads=" << threads
                                                  << " filtered=" << filtered);
                cpu::Frame frame(count);
                frame.box = cpu::Box::orthorhombic(7.6, 20.0, 20.0);
                frame.span(kAtom_x)[0] = 0.2;
                frame.span(kAtom_x)[1] = 3.8;
                for (int i = 2; i < count; ++i)
                    frame.span(kAtom_x)[i] = -3.8;
                cpu::md::PairList pairs;
                pairs.inum = 1;
                pairs.n_rows = count;
                pairs.ilist = {0};
                pairs.numneigh.assign(count, 0);
                pairs.numneigh[0] = count - 1;
                pairs.firstneigh.assign(count, count - 1);
                pairs.firstneigh[0] = 0;
                for (int i = 1; i < count; ++i)
                    pairs.neighbors.push_back(i);
                pairs.total_pairs = count - 1;
                cpu::Context context;
                context.md.pair_list = &pairs;
                cpu::EAM eam{pot, cutoff};
                eam.build({context, frame});
                eam.launch({context, frame, 0, 0, true, true});
                EXPECT_NEAR(frame.meta.get(kEnergy), eam_energy, 1e-11);
                EXPECT_NEAR(frame.span(kAtom_fx)[0], eam_derivative, 1e-11);
                EXPECT_NEAR(frame.meta.get(kStress)[0], -r * eam_derivative,
                            1e-10);
                for (int i = 2; i < count; ++i)
                    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[i], 0.0);
                eam.destroy({context, frame});

                for (const auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
                    std::fill_n(frame.span(column).data(), count, 0.0);
                frame.meta.get(kEnergy) = frame.meta.get(kEnergyPair) = 0.0;
                frame.meta.get(kStress).fill(0.0);
                cpu::PairLJCut lj{0.1, 2.0, cutoff};
                if (filtered)
                    lj.set_special_bonds(0.0, 0.0, 0.5);
                lj.build({context, frame});
                lj.launch({context, frame, 0, 0, true, true});
                EXPECT_NEAR(frame.meta.get(kEnergy), lj_energy, 1e-12);
                EXPECT_NEAR(frame.span(kAtom_fx)[0], lj_derivative, 1e-12);
                EXPECT_NEAR(frame.meta.get(kStress)[0], -r * lj_derivative,
                            1e-12);
                for (int i = 2; i < count; ++i)
                    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[i], 0.0);
                lj.destroy({context, frame});
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
#endif
}

TEST(PairLJCut, TriclinicBatchAndTailUseExplicitImageRows) {
    cpu::Frame frame(10);
    Mat3 cell = Mat3::diagonal(8.0, 9.0, 10.0);
    cell(0, 1) = 1.2;
    frame.box = cpu::Box(cell);
    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.ilist = {0};
    pairs.numneigh.assign(10, 0);
    pairs.numneigh[0] = 9;
    pairs.firstneigh.assign(10, 9);
    pairs.firstneigh[0] = 0;
    constexpr double cutoff = 4.5;
    const double c6 = std::pow(2.0 / cutoff, 6);
    double energy = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 1; j < 10; ++j) {
        pairs.neighbors.push_back(j);
        frame.span(kAtom_x)[j] = 7.0 + 0.01 * j;
        frame.span(kAtom_y)[j] = 6.0;
        frame.span(kAtom_z)[j] = 1.0;
        double dx = -frame.span(kAtom_x)[j], dy = -6.0, dz = -1.0;
        frame.box.min_image(dx, dy, dz);
        frame.span(kAtom_x)[j] = -dx;
        frame.span(kAtom_y)[j] = -dy;
        frame.span(kAtom_z)[j] = -dz;
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double s6 = std::pow(4.0 / r2, 3);
        energy += 0.4 * (s6 * s6 - s6 - c6 * c6 + c6);
        const double force = 2.4 * (2 * s6 * s6 - s6) / r2;
        fx += force * dx;
        fy += force * dy;
        fz += force * dz;
    }
    pairs.total_pairs = 9;
    cpu::Context context;
    context.md.pair_list = &pairs;
    cpu::PairLJCut lj{0.1, 2.0, cutoff};
    lj.build({context, frame});
    lj.launch({context, frame, 0, 0});
    EXPECT_NEAR(frame.meta.get(kEnergy), energy, 1e-12);
    EXPECT_NEAR(frame.span(kAtom_fx)[0], fx, 1e-12);
    EXPECT_NEAR(frame.span(kAtom_fy)[0], fy, 1e-12);
    EXPECT_NEAR(frame.span(kAtom_fz)[0], fz, 1e-12);
    lj.destroy({context, frame});
}

#ifdef _OPENMP
TEST(EAM, SerializedNestedTeamDoesNotReuseStaleThreadBuffers) {
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    const int previous_levels = omp_get_max_active_levels();
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    omp_set_max_active_levels(1);
    cpu::Frame frame = copper(3.61, 2, 0.12);
    const double cutoff = 5.5;
    ATV::test::PeriodicNeighborFixture fixture(frame, cutoff + 0.3);
    auto& context = fixture.context;
    cpu::md::NeighborList<> neighbors{cutoff, 0.3, 1, 0, true};
    neighbors.build({context, frame});
    neighbors.launch({context, frame, 0, 0});
    auto potential =
        cpu::EAM{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"}, cutoff};
    potential.build({context, frame});
    potential.launch({context, frame, 0, 0, true, true});
    // Direct launches outside a slot walk land the shared
    // per-thread accumulators themselves, as the Runtime does.
    context.md.chunk_forces->flush();
    context.md.comm->reverse_comm({context, frame, 0, 0});
    const cpu::Frame expected = frame;
    for (const auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
        std::fill_n(frame.span(column).data(), frame.n_total(), 0.0);
    frame.meta.get(kEnergy) = frame.meta.get(kEnergyPair) = 0.0;
    for (int d = 0; d < 6; ++d)
        frame.meta.get(kStress)[d] = 0.0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            potential.launch({context, frame, 0, 0, true, true});
            // Direct launches outside a slot walk land the shared
            // per-thread accumulators themselves, as the Runtime does.
            context.md.chunk_forces->flush();
            context.md.comm->reverse_comm({context, frame, 0, 0});
        }
    }
    EXPECT_NEAR(frame.meta.get(kEnergy), expected.meta.get(kEnergy), 1e-10);
    for (const auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < frame.nlocal(); ++i)
            EXPECT_NEAR(frame.span(column)[i], expected.span(column)[i], 1e-10);
    for (int d = 0; d < 6; ++d)
        EXPECT_NEAR(frame.meta.get(kStress)[d], expected.meta.get(kStress)[d],
                    1e-9);
    const cpu::Frame before_energy = frame;
    frame.meta.get(kEnergy) = frame.meta.get(kEnergyPair) = 0.0;
    potential.launch({context, frame, 0, 0, false, false});
    // Direct launches outside a slot walk land the shared
    // per-thread accumulators themselves, as the Runtime does.
    context.md.chunk_forces->flush();
    EXPECT_NEAR(frame.meta.get(kEnergy), expected.meta.get(kEnergy), 1e-10);
    for (const auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < frame.nlocal(); ++i)
            EXPECT_DOUBLE_EQ(frame.span(column)[i],
                             before_energy.span(column)[i]);
    for (int d = 0; d < 6; ++d)
        EXPECT_DOUBLE_EQ(frame.meta.get(kStress)[d],
                         before_energy.meta.get(kStress)[d]);
    potential.destroy({context, frame});
    neighbors.destroy({context, frame});
    omp_set_max_active_levels(previous_levels);
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
}
#endif

TEST(PeriodicImages, TriclinicLJAndEAMMatchExplicitImageEnergyForceVirial) {
    cpu::Frame source(2);
    Mat3 cell = Mat3::diagonal(3.2, 3.4, 3.6);
    cell(0, 1) = 0.7;
    cell(1, 2) = 0.4;
    cell(2, 0) = 0.1;
    source.box = cpu::Box(cell);
    const Vec3 fractions[] = {{0.11, 0.23, 0.3}, {0.5, 0.61, 0.67}};
    Vec3 positions[2];
    for (int i = 0; i < 2; ++i) {
        positions[i] = source.box.to_cartesian(fractions[i]);
        source.span(kAtom_x)[i] = positions[i][0];
        source.span(kAtom_y)[i] = positions[i][1];
        source.span(kAtom_z)[i] = positions[i][2];
        source.span(kAtom_tag)[i] = i + 1;
    }
    constexpr double cutoff = 3.9, epsilon = 0.2;
    const cpu::EAMPotential ep(molrs::Element::Cu, "sutton-chen");
    struct Image {
        int i, j;
        Vec3 delta;
        double r;
    };
    std::vector<Image> images;
    double rho[2]{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int nx = -2; nx <= 2; ++nx)
                for (int ny = -2; ny <= 2; ++ny)
                    for (int nz = -2; nz <= 2; ++nz) {
                        if (i == j && nx == 0 && ny == 0 && nz == 0)
                            continue;
                        const Vec3 delta =
                            positions[i] - positions[j]
                            - source.box.to_cartesian(
                                Vec3{double(nx), double(ny), double(nz)});
                        const double r = delta.norm();
                        if (r >= cutoff)
                            continue;
                        images.push_back({i, j, delta, r});
                        rho[i] += ep.f(r);
                    }
    for (bool eam : {false, true}) {
        SCOPED_TRACE(eam ? "EAM" : "LJ");
        double expected_energy = eam ? ep.F(rho[0]) + ep.F(rho[1]) : 0.0;
        Vec3 expected_force[2];
        double expected_stress[6]{};
        for (const auto& image : images) {
            const double r = image.r;
            const double invr6 = std::pow(1.0 / r, 6);
            const double cut6 = std::pow(1.0 / cutoff, 6);
            const double energy =
                eam ? ep.phi(r) - ep.phi(cutoff)
                    : 4 * epsilon
                          * (invr6 * invr6 - invr6 - cut6 * cut6 + cut6);
            const double scalar =
                eam ? -(ep.dphi(r)
                        + (ep.dF(rho[image.i]) + ep.dF(rho[image.j]))
                              * ep.df(r))
                          / r
                    : 24 * epsilon * (2 * invr6 * invr6 - invr6) / (r * r);
            const Vec3 force = image.delta * scalar;
            expected_energy += 0.5 * energy;
            expected_force[image.i] = expected_force[image.i] + force;
            const double dx = image.delta[0], dy = image.delta[1],
                         dz = image.delta[2];
            const double tensor[] = {dx * force[0], dy * force[1],
                                     dz * force[2], dx * force[1],
                                     dx * force[2], dy * force[2]};
            for (int k = 0; k < 6; ++k)
                expected_stress[k] += 0.5 * tensor[k];
        }
        for (bool derive : {false, true}) {
            SCOPED_TRACE(derive ? "derived half" : "direct half");
            auto frame = source;
            ATV::test::PeriodicNeighborFixture fixture(frame, cutoff + 0.2);
            auto& context = fixture.context;
            cpu::md::CellList cells(cutoff, 0.2);
            cells.init_extended(frame, cutoff + 0.2, context.md.dd,
                                frame.capacity());
            cells.build(frame);
            cpu::md::PairList half;
            if (derive) {
                const auto full = cells.build_full(frame);
                cells.derive_half(frame, frame.n_total(), full, half,
                                  frame.nlocal());
            } else {
                half = cells.build_half(frame);
            }
            context.md.pair_list = &half;
            if (eam) {
                cpu::EAM potential(ep, cutoff);
                potential.build({context, frame});
                potential.launch({context, frame, 0, 0, true, true});
                context.md.comm->reverse_comm({context, frame, 0, 0});
                potential.destroy({context, frame});
            } else {
                cpu::PairLJCut potential(epsilon, 1.0, cutoff);
                potential.build({context, frame});
                potential.launch({context, frame, 0, 0, true, true});
                context.md.comm->reverse_comm({context, frame, 0, 0});
                potential.destroy({context, frame});
            }
            EXPECT_NEAR(frame.meta.get(kEnergy), expected_energy, 1e-10);
            for (int i = 0; i < 2; ++i) {
                EXPECT_NEAR(frame.span(kAtom_fx)[i], expected_force[i][0],
                            1e-10);
                EXPECT_NEAR(frame.span(kAtom_fy)[i], expected_force[i][1],
                            1e-10);
                EXPECT_NEAR(frame.span(kAtom_fz)[i], expected_force[i][2],
                            1e-10);
            }
            for (int k = 0; k < 6; ++k)
                EXPECT_NEAR(frame.meta.get(kStress)[k], expected_stress[k],
                            1e-10);
        }
    }
}
