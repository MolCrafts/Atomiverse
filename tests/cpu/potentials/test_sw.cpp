/// @file tests/cpu/potentials/test_sw.cpp
/// @brief Unit: Stillinger-Weber — lookup, the angular term, and the FD force
///        oracle for a genuine three-body force.

#include <cmath>
#include <limits>
#include <vector>

#include "../dynamics/periodic_neighbor_fixture.h"
#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/sw.h"

using namespace ATV;

namespace {

/// Diamond-cubic silicon, optionally distorted — several call sites below.
cpu::Frame silicon(double a0, int n_cell, double jitter) {
    const double b[8][3] = {
        {0, 0, 0},     {0.25, 0.25, 0.25}, {0.5, 0.5, 0}, {0.75, 0.75, 0.25},
        {0.5, 0, 0.5}, {0.75, 0.25, 0.75}, {0, 0.5, 0.5}, {0.25, 0.75, 0.75}};
    const int n = 8 * n_cell * n_cell * n_cell;
    cpu::Frame f(n);
    int idx = 0;
    for (int ix = 0; ix < n_cell; ++ix)
        for (int iy = 0; iy < n_cell; ++iy)
            for (int iz = 0; iz < n_cell; ++iz)
                for (int k = 0; k < 8; ++k) {
                    const double j = jitter * std::sin(double(11 * idx + 5));
                    f.span(ATV::kAtom_x)[idx] = (double(ix) + b[k][0]) * a0 + j;
                    f.span(ATV::kAtom_y)[idx] =
                        (double(iy) + b[k][1]) * a0 - 0.6 * j;
                    f.span(ATV::kAtom_z)[idx] =
                        (double(iz) + b[k][2]) * a0 + 0.4 * j;
                    f.span(ATV::kAtom_mass)[idx] = 28.0855;
                    f.span(ATV::kAtom_type)[idx] = 14;
                    ++idx;
                }
    f.box = ATV::cpu::Box::cubic(a0 * n_cell);
    return f;
}

/// Single-point SW energy — one call per FD displacement, i.e. many.
double energy_of(cpu::Frame& f) {
    const cpu::SWPotential p(molrs::Element::Si, "stillinger-weber");
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{p.cutoff(), 0.3, 1, 0, true});
    eval.on(evaluate::Phase::evaluate).add(cpu::StillingerWeber{p});
    eval.enable_timing(false);
    eval.build();
    eval.run(f);
    return f.meta.get(ATV::kEnergy);
}

} // namespace

TEST(SW, LookupThrowsOnUnsupported) {
    EXPECT_NO_THROW(cpu::SWPotential(molrs::Element::Si, "stillinger-weber"));
    EXPECT_THROW(cpu::SWPotential(molrs::Element::Si, "nope"),
                 std::out_of_range);
    EXPECT_THROW(cpu::SWPotential(molrs::Element::Cu, "stillinger-weber"),
                 std::out_of_range);
}

/// Both terms vanish identically at the cutoff a*sigma — SW has no discontinuity
/// to shift away, unlike a power-law potential.
TEST(SW, TermsVanishAtCutoff) {
    const cpu::SWPotential p(molrs::Element::Si, "stillinger-weber");
    const double rc = p.cutoff();
    EXPECT_NEAR(rc, 1.80 * 2.0951, 1e-12);
    EXPECT_EQ(p.phi2(rc), 0.0);
    EXPECT_EQ(p.dphi2(rc), 0.0);
    EXPECT_EQ(p.g(rc), 0.0);
    // Approaching from inside, the exponential drives them smoothly to zero.
    EXPECT_LT(std::fabs(p.phi2(rc - 1e-3)), 1e-6);
    EXPECT_LT(std::fabs(p.g(rc - 1e-3)), 1e-6);
}

TEST(SW, TwoBodyDerivativeMatchesFD) {
    const cpu::SWPotential p(molrs::Element::Si, "stillinger-weber");
    const double h = 1e-7;
    for (double r : {2.0, 2.35, 2.8, 3.2, 3.6}) {
        const double fd = (p.phi2(r + h) - p.phi2(r - h)) / (2 * h);
        EXPECT_NEAR(p.dphi2(r), fd, 1e-5 * std::fabs(fd) + 1e-7) << "r=" << r;
        const double gfd = (p.g(r + h) - p.g(r - h)) / (2 * h);
        EXPECT_NEAR(p.dg(r), gfd, 1e-5 * std::fabs(gfd) + 1e-9) << "r=" << r;
    }
}

/// The angular term is minimized at the tetrahedral angle — this is the physics
/// SW was built to encode, and it is what makes silicon prefer the diamond
/// structure over a close-packed one.
TEST(SW, AngularTermIsMinimizedAtTetrahedralAngle) {
    const cpu::SWPotential p(molrs::Element::Si, "stillinger-weber");
    const double cos0 = p.cos_theta0();
    EXPECT_NEAR(cos0, -1.0 / 3.0, 1e-12);
    const double theta0 = std::acos(cos0) * 180.0 / M_PI;
    EXPECT_NEAR(theta0, 109.47, 0.01); // the tetrahedral angle

    // (cos(theta) - cos0)^2 is zero at theta0 and positive on either side.
    auto ang = [&](double deg) {
        const double c = std::cos(deg * M_PI / 180.0);
        return (c - cos0) * (c - cos0);
    };
    EXPECT_NEAR(ang(109.47), 0.0, 1e-6);
    EXPECT_GT(ang(90.0), 1e-3);
    EXPECT_GT(ang(120.0), 1e-3);
}

/// THE GATE: the three-body force — including the angular derivative, where the
/// signs are easy to get wrong — is the gradient of the reported energy.
TEST(SW, AnalyticForceMatchesFiniteDifference) {
    cpu::Frame f = silicon(5.431, 2, /*jitter=*/0.15); // 64 atoms, distorted
    const int n = f.nlocal();

    const cpu::SWPotential p(molrs::Element::Si, "stillinger-weber");
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{p.cutoff(), 0.3, 1, 0, true});
    eval.on(evaluate::Phase::evaluate).add(cpu::StillingerWeber{p});
    eval.enable_timing(false);
    eval.build();
    eval.run(f);

    std::vector<double> fx(n), fy(n), fz(n);
    for (int i = 0; i < n; ++i) {
        fx[i] = f.span(ATV::kAtom_fx)[i];
        fy[i] = f.span(ATV::kAtom_fy)[i];
        fz[i] = f.span(ATV::kAtom_fz)[i];
    }

    const double h = 1.0e-5;
    const double force_tol = 1.0e-5;
    double worst = 0.0;
    for (int i : {0, 9, 33, 63}) {
        for (int d = 0; d < 3; ++d) {
            auto coord = (d == 0)   ? f.span(ATV::kAtom_x)
                         : (d == 1) ? f.span(ATV::kAtom_y)
                                    : f.span(ATV::kAtom_z);
            const double x0 = coord[i];

            coord[i] = x0 + h;
            cpu::Frame plus = f;
            const double e_plus = energy_of(plus);

            coord[i] = x0 - h;
            cpu::Frame minus = f;
            const double e_minus = energy_of(minus);

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

/// Newton's third law survives the triplet decomposition (the vertex force is
/// derived from translation invariance, so this also checks the two leg forces).
TEST(SW, ForcesSumToZero) {
    cpu::Frame f = silicon(5.431, 2, 0.15);
    const cpu::SWPotential p(molrs::Element::Si, "stillinger-weber");
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{p.cutoff(), 0.3, 1, 0, true});
    eval.on(evaluate::Phase::evaluate).add(cpu::StillingerWeber{p});
    eval.enable_timing(false);
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

TEST(SW, FullListPairOwnershipPreservesForceAndStressRequests) {
    const cpu::SWPotential p{molrs::Element::Si, "stillinger-weber"};
    constexpr double r = 2.35;
    constexpr double h = 1e-6;
    const double derivative = (p.phi2(r + h) - p.phi2(r - h)) / (2 * h);
    const TagInt top = std::numeric_limits<TagInt>::max();
    // Exercise both parity branches, both tag orders, and tags whose sum
    // would overflow. A two-atom full list has no angular contribution.
    for (const auto tags : {std::array<TagInt, 2>{top - 1, top},
                            std::array<TagInt, 2>{top, top - 1},
                            std::array<TagInt, 2>{top - 2, top},
                            std::array<TagInt, 2>{top, top - 2}}) {
        for (bool forces : {false, true}) {
            for (bool stress : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "tags=" << tags[0] << ',' << tags[1]
                             << " forces=" << forces << " stress=" << stress);
                cpu::Frame frame(2);
                frame.box = cpu::Box::cubic(20.0);
                frame.span(kAtom_x)[0] = 1.0;
                frame.span(kAtom_x)[1] = 1.0 + r;
                for (int i = 0; i < 2; ++i) {
                    frame.span(kAtom_tag)[i] = tags[i];
                    frame.span(kAtom_fx)[i] = 7.0;
                    frame.span(kAtom_fy)[i] = 8.0;
                    frame.span(kAtom_fz)[i] = 9.0;
                }
                for (int d = 0; d < 6; ++d)
                    frame.meta.get(kStress)[d] = 10.0 + d;
                ATV::test::PeriodicNeighborFixture fixture(frame,
                                                           p.cutoff() + 0.3);
                auto& context = fixture.context;
                cpu::md::NeighborList<> neighbors{p.cutoff(), 0.3, 1, 0, true};
                neighbors.configure_lists(false, true);
                neighbors.build({context, frame});
                neighbors.launch({context, frame, 0, 0});
                cpu::StillingerWeber potential{p};
                potential.build({context, frame});
                potential.launch({context, frame, 0, 0, forces, stress});
                EXPECT_NEAR(frame.meta.get(kEnergy), p.phi2(r), 1e-12);
                EXPECT_NEAR(frame.span(kAtom_fx)[0],
                            7.0 + (forces ? derivative : 0.0), 1e-8);
                EXPECT_NEAR(frame.span(kAtom_fx)[1],
                            7.0 - (forces ? derivative : 0.0), 1e-8);
                for (int i = 0; i < 2; ++i) {
                    EXPECT_DOUBLE_EQ(frame.span(kAtom_fy)[i], 8.0);
                    EXPECT_DOUBLE_EQ(frame.span(kAtom_fz)[i], 9.0);
                }
                EXPECT_NEAR(frame.meta.get(kStress)[0],
                            10.0 - (stress ? r * derivative : 0.0), 1e-8);
                for (int d = 1; d < 6; ++d)
                    EXPECT_DOUBLE_EQ(frame.meta.get(kStress)[d], 10.0 + d);
                potential.destroy({context, frame});
                neighbors.destroy({context, frame});
            }
        }
    }
}

#ifdef _OPENMP
TEST(SW, SerializedNestedTeamDoesNotReuseStaleThreadBuffers) {
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    const int previous_levels = omp_get_max_active_levels();
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    omp_set_max_active_levels(1);
    cpu::Frame frame = silicon(5.431, 2, 0.15);
    const double cutoff = 3.77118;
    ATV::test::PeriodicNeighborFixture fixture(frame, cutoff + 0.3);
    auto& context = fixture.context;
    cpu::md::NeighborList<> neighbors{cutoff, 0.3, 1, 0, true};
    neighbors.configure_lists(false, true);
    neighbors.build({context, frame});
    neighbors.launch({context, frame, 0, 0});
    auto potential = cpu::StillingerWeber{
        cpu::SWPotential{molrs::Element::Si, "stillinger-weber"}};
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
    // The angular derivatives are also needed for stress-only requests, while
    // force-only requests must preserve an already populated stress tensor.
    for (bool forces : {false, true}) {
        frame = before_energy;
        frame.meta.get(kEnergy) = frame.meta.get(kEnergyPair) = 0.0;
        if (forces)
            for (const auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
                std::fill_n(frame.span(column).data() + frame.nlocal(),
                            frame.nghost(), 0.0);
        potential.launch({context, frame, 0, 0, forces, !forces});
        context.md.chunk_forces->flush();
        if (forces)
            context.md.comm->reverse_comm({context, frame, 0, 0});
        EXPECT_NEAR(frame.meta.get(kEnergy), expected.meta.get(kEnergy), 1e-10);
        for (const auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
            for (int i = 0; i < frame.nlocal(); ++i)
                EXPECT_NEAR(frame.span(column)[i],
                            before_energy.span(column)[i]
                                + (forces ? expected.span(column)[i] : 0.0),
                            1e-10);
        for (int d = 0; d < 6; ++d)
            EXPECT_NEAR(frame.meta.get(kStress)[d],
                        before_energy.meta.get(kStress)[d]
                            + (forces ? 0.0 : expected.meta.get(kStress)[d]),
                        1e-9);
    }
    potential.destroy({context, frame});
    neighbors.destroy({context, frame});
    omp_set_max_active_levels(previous_levels);
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
}
#endif

TEST(StillingerWeber, TriclinicSelfImagesKeepAngularLegsAndVirial) {
    const cpu::SWPotential potential(molrs::Element::Si, "stillinger-weber");
    Mat3 cell = Mat3::diagonal(3.2, 3.3, 3.4);
    cell(0, 1) = 0.4;
    cell(1, 2) = 0.2;
    cell(2, 0) = 0.1;
    auto oracle = [&](const Mat3& matrix) {
        const cpu::Box box(matrix);
        std::vector<Vec3> legs;
        double energy = 0.0;
        for (int x = -2; x <= 2; ++x)
            for (int y = -2; y <= 2; ++y)
                for (int z = -2; z <= 2; ++z) {
                    if (x == 0 && y == 0 && z == 0)
                        continue;
                    const Vec3 leg =
                        box.to_cartesian(Vec3{double(x), double(y), double(z)});
                    if (leg.norm() >= potential.cutoff())
                        continue;
                    energy += 0.5 * potential.phi2(leg.norm());
                    legs.push_back(leg);
                }
        for (std::size_t j = 0; j < legs.size(); ++j)
            for (std::size_t k = j + 1; k < legs.size(); ++k) {
                const double rj = legs[j].norm(), rk = legs[k].norm();
                const double cosine = legs[j].dot(legs[k]) / (rj * rk);
                const double delta = cosine - potential.cos_theta0();
                energy += potential.lambda_eps() * potential.g(rj)
                          * potential.g(rk) * delta * delta;
            }
        return energy;
    };
    cpu::Frame frame(1);
    frame.box = cpu::Box(cell);
    const Vec3 position = frame.box.to_cartesian(Vec3{0.3, 0.4, 0.2});
    frame.span(kAtom_x)[0] = position[0];
    frame.span(kAtom_y)[0] = position[1];
    frame.span(kAtom_z)[0] = position[2];
    frame.span(kAtom_tag)[0] = 7;
    ATV::test::PeriodicNeighborFixture fixture(frame, potential.cutoff() + 0.2);
    auto& context = fixture.context;
    cpu::md::CellList cells(potential.cutoff(), 0.2);
    cells.init_extended(frame, potential.cutoff() + 0.2, context.md.dd,
                        frame.capacity());
    cells.build(frame);
    auto pairs = cells.build_full(frame);
    ASSERT_GT(pairs.total_pairs, 2);
    context.md.full_pair_list = &pairs;
    cpu::StillingerWeber sw(potential);
    sw.build({context, frame});
    sw.launch({context, frame, 0, 0, true, true});
    context.md.comm->reverse_comm({context, frame, 0, 0});
    EXPECT_NEAR(frame.meta.get(kEnergy), oracle(cell), 1e-11);
    EXPECT_NEAR(frame.span(kAtom_fx)[0], 0.0, 1e-11);
    EXPECT_NEAR(frame.span(kAtom_fy)[0], 0.0, 1e-11);
    EXPECT_NEAR(frame.span(kAtom_fz)[0], 0.0, 1e-11);
    constexpr int first[] = {0, 1, 2, 0, 0, 1};
    constexpr int second[] = {0, 1, 2, 1, 2, 2};
    constexpr double h = 1e-6;
    for (int k = 0; k < 6; ++k) {
        Mat3 plus = cell, minus = cell;
        for (int col = 0; col < 3; ++col) {
            plus(first[k], col) += h * cell(second[k], col);
            minus(first[k], col) -= h * cell(second[k], col);
        }
        const double expected = -(oracle(plus) - oracle(minus)) / (2 * h);
        EXPECT_NEAR(frame.meta.get(kStress)[k], expected, 1e-6)
            << "component " << k;
    }
    sw.destroy({context, frame});
}
