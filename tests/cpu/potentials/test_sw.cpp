/// @file tests/cpu/potentials/test_sw.cpp
/// @brief Unit: Stillinger-Weber — lookup, the angular term, and the FD force
///        oracle for a genuine three-body force.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/driver/evaluate.h"
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
    f.simbox = ATV::cpu::Box::cubic(a0 * n_cell);
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
