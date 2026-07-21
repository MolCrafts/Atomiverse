/// @file tests/cpu/gpw/test_cpu_gpw_mt_poisson.cpp
/// @brief Isolated (truncated-Coulomb) Poisson: matches the analytic isolated
///        Gaussian potential/energy and is box-size independent.
///
/// A normalized Gaussian charge rho(r) = (2 pi s^2)^{-3/2} exp(-r^2/2s^2) has the
/// exact open-boundary potential V(r) = erf(r/(sqrt2 s))/r and self-energy
/// E = 1/(2 s sqrt(pi)). The truncated-Coulomb solver must reproduce both, while
/// the bare periodic solver (G=0 dropped) gives a box-dependent answer.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"

using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::PoissonResult;
using ATV::cpu::gpw::solve_poisson;
using ATV::cpu::gpw::solve_poisson_mt;
using ATV::cpu::gpw::UniformGrid;

namespace {

// Normalized Gaussian charge of width sigma centered in the box, on the grid.
std::vector<double> gaussian_density(const UniformGrid& g, double sigma) {
    const double c[3] = {g.origin[0] + 0.5 * g.L[0], g.origin[1] + 0.5 * g.L[1],
                         g.origin[2] + 0.5 * g.L[2]};
    const double s2 = sigma * sigma;
    const double norm = std::pow(2.0 * M_PI * s2, -1.5);
    std::vector<double> rho(static_cast<std::size_t>(g.npts()));
    for (long i = 0; i < g.npts(); ++i) {
        double r[3];
        g.point(i, r);
        const double d2 = (r[0] - c[0]) * (r[0] - c[0])
                          + (r[1] - c[1]) * (r[1] - c[1])
                          + (r[2] - c[2]) * (r[2] - c[2]);
        rho[static_cast<std::size_t>(i)] = norm * std::exp(-d2 / (2.0 * s2));
    }
    return rho;
}

double analytic_self_energy(double sigma) {
    return 1.0 / (2.0 * sigma * std::sqrt(M_PI));
}

} // namespace

TEST(GpwMtPoisson, MatchesAnalyticGaussianPotential) {
    const double L = 16.0, sigma = 1.0;
    const double cell[3] = {L, L, L};
    UniformGrid g = make_uniform_grid(cell, 200.0);
    const std::vector<double> rho = gaussian_density(g, sigma);

    const PoissonResult mt = solve_poisson_mt(g, rho);

    // Potential along +x from the center grid point (n/2,n/2,n/2). Compare to the
    // analytic isolated V(r)=erf(r/(sqrt2 sigma))/r in the smooth region.
    const int n = g.n[0];
    const double h = g.L[0] / n;
    const int half = n / 2;
    double max_rel = 0.0;
    for (int di = 2; di < n / 2; ++di) {
        const double r = di * h;
        if (r > 0.6 * 0.5 * L)
            break; // stay well inside the cutoff sphere
        const long idx = (static_cast<long>(half + di) * n + half) * n + half;
        const double v_num = mt.v_r[static_cast<std::size_t>(idx)];
        const double v_ana = std::erf(r / (std::sqrt(2.0) * sigma)) / r;
        max_rel = std::max(max_rel, std::abs(v_num - v_ana) / std::abs(v_ana));
    }
    EXPECT_LT(max_rel, 5e-3) << "max rel potential error = " << max_rel;

    // Self-energy.
    const double e_ana = analytic_self_energy(sigma);
    EXPECT_NEAR(mt.e_hartree, e_ana, 5e-3 * e_ana)
        << "E_mt=" << mt.e_hartree << " analytic=" << e_ana;
}

TEST(GpwMtPoisson, BoxSizeIndependentUnlikePeriodic) {
    const double sigma = 1.0;
    const double e_ana = analytic_self_energy(sigma);

    double e_mt[2], e_per[2];
    int k = 0;
    for (double L : {16.0, 24.0}) {
        const double cell[3] = {L, L, L};
        UniformGrid g = make_uniform_grid(cell, 200.0);
        const std::vector<double> rho = gaussian_density(g, sigma);
        e_mt[k] = solve_poisson_mt(g, rho).e_hartree;
        e_per[k] = solve_poisson(g, rho).e_hartree; // periodic, neutralizing bg
        ++k;
    }

    // MT energy is box-independent and matches the analytic isolated value.
    EXPECT_NEAR(e_mt[0], e_ana, 5e-3 * e_ana);
    EXPECT_NEAR(e_mt[1], e_ana, 5e-3 * e_ana);
    EXPECT_NEAR(e_mt[0], e_mt[1], 5e-3 * e_ana) << "MT drifts with box size";

    // The periodic solver drifts with box size and is the wrong (non-isolated)
    // value — confirming MT is doing something different and correct.
    EXPECT_GT(std::abs(e_per[0] - e_per[1]), std::abs(e_mt[0] - e_mt[1]));
}
