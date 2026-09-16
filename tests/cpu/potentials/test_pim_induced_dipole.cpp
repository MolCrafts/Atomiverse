/// @file tests/cpu/potentials/test_pim_induced_dipole.cpp
/// @brief Unit tests for cpu::PimInducedDipole (SCC, Thole/TT, FD forces).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"

using namespace ATV;

namespace {

cpu::Frame ion_pair(double r, double q0, double q1, int t0, int t1) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = q0;
    f.span(kAtom_charge)[1] = q1;
    f.span(kAtom_type)[0] = t0;
    f.span(kAtom_type)[1] = t1;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
    f.span(kAtom_fx)[1] = f.span(kAtom_fy)[1] = f.span(kAtom_fz)[1] = 0;
    f.box = cpu::Box::cubic(50.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

struct PolResult {
    double e_pol = 0;
    double mu[6]{};
    double fx[2]{}, fy[2]{}, fz[2]{};
    int iters = 0;
};

PolResult run_pim(double r, double thole_a, double tt_b,
                  const std::vector<double>& alpha, double q0 = 1.0,
                  double q1 = -1.0) {
    auto f = ion_pair(r, q0, q1, 1, 2);
    cpu::Context ctx;
    cpu::PimInducedDipole pim(20.0, alpha, thole_a, 1e-12, 200);
    pim.tt_b = tt_b;
    pim.build({ctx, f});
    pim.launch({ctx, f, 0, 0});
    PolResult out;
    out.e_pol = pim.last_pol_energy();
    out.iters = pim.last_pol_iters();
    EXPECT_TRUE(pim.last_pol_converged());
    const auto& mu = pim.last_mu();
    EXPECT_EQ(mu.size(), 6u);
    if (mu.size() >= 6u) {
        for (int k = 0; k < 6; ++k)
            out.mu[k] = mu[static_cast<std::size_t>(k)];
    }
    for (int i = 0; i < 2; ++i) {
        out.fx[i] = f.span(kAtom_fx)[i];
        out.fy[i] = f.span(kAtom_fy)[i];
        out.fz[i] = f.span(kAtom_fz)[i];
    }
    EXPECT_NEAR(f.meta.get(kEnergy), out.e_pol, 1e-14);
    return out;
}

/// Closed-form undamped two-site SCC along x (no Thole).
/// Sites at 0 and r on x; E_i = q_j (r_i−r_j)/|r|³ ⇒
/// E0_x = −q1/r², E1_x = +q0/r²; T_xx = 2/r³.
void undamped_two_site(double r, double q0, double q1, double a0, double a1,
                       double& e_pol, double& mu0, double& mu1) {
    const double E0 = -q1 / (r * r); // at site 0
    const double E1 = q0 / (r * r);  // at site 1
    const double T = 2.0 / (r * r * r);
    // μ0 = a0 (E0 + T μ1); μ1 = a1 (E1 + T μ0)
    const double den = 1.0 - a0 * a1 * T * T;
    mu0 = (a0 * E0 + a0 * a1 * T * E1) / den;
    mu1 = (a1 * E1 + a1 * a0 * T * E0) / den;
    e_pol = units::coulomb * (-0.5 * (mu0 * E0 + mu1 * E1));
}

} // namespace

TEST(PimInducedDipole, UndampedTwoSiteMatchesClosedForm) {
    const double r = 3.0, q0 = 1.0, q1 = -1.0, a0 = 1.0, a1 = 1.5;
    double e_ref, mu0, mu1;
    undamped_two_site(r, q0, q1, a0, a1, e_ref, mu0, mu1);

    auto out = run_pim(r, /*thole=*/0.0, /*tt=*/0.0, {0.0, a0, a1});
    EXPECT_NEAR(out.e_pol, e_ref, 1e-10);
    EXPECT_NEAR(out.mu[0], mu0, 1e-10);
    EXPECT_NEAR(out.mu[1], 0.0, 1e-12);
    EXPECT_NEAR(out.mu[2], 0.0, 1e-12);
    EXPECT_NEAR(out.mu[3], mu1, 1e-10);
    EXPECT_LT(out.e_pol, 0.0);
}

TEST(PimInducedDipole, OnePolUndampedAnalyticForce) {
    // Site 0 polarizable, site 1 charge-only: μ0=α E0, E0=1/r² (q=±1),
    // U=-½ k α/r⁴, F0x = 2 k α / r⁵, F1x = −F0x.
    const double r = 3.0, a0 = 1.0;
    auto out = run_pim(r, 0.0, 0.0, {0.0, a0, 0.0});
    const double k = units::coulomb;
    const double F0 = 2.0 * k * a0 / std::pow(r, 5);
    const double U = -0.5 * k * a0 / std::pow(r, 4);
    EXPECT_NEAR(out.e_pol, U, 1e-12);
    EXPECT_NEAR(out.fx[0], F0, 1e-10);
    EXPECT_NEAR(out.fx[1], -F0, 1e-10);
}

TEST(PimInducedDipole, TholeChangesEnergy) {
    auto und = run_pim(3.0, 0.0, 0.0, {0.0, 1.0, 1.5});
    auto th = run_pim(3.0, 0.39, 0.0, {0.0, 1.0, 1.5});
    EXPECT_NE(und.e_pol, th.e_pol);
    EXPECT_TRUE(std::isfinite(th.e_pol));
    EXPECT_LT(th.e_pol, 0.0);
    // Thole softens short-range induction → |E_pol| typically smaller
    EXPECT_LT(std::fabs(th.e_pol), std::fabs(und.e_pol));
}

TEST(PimInducedDipole, TangToenniesPathFinite) {
    auto out = run_pim(3.0, /*thole=*/0.0, /*tt_b=*/1.5, {0.0, 1.0, 1.5});
    EXPECT_TRUE(std::isfinite(out.e_pol));
    EXPECT_LT(out.e_pol, 0.0);
    EXPECT_GT(out.iters, 0);
}

TEST(PimInducedDipole, ZeroAlphaGivesZeroPol) {
    auto out = run_pim(3.0, 0.39, 0.0, {0.0, 0.0, 0.0});
    EXPECT_NEAR(out.e_pol, 0.0, 1e-15);
    EXPECT_NEAR(out.fx[0], 0.0, 1e-15);
    EXPECT_NEAR(out.fx[1], 0.0, 1e-15);
}

TEST(PimInducedDipole, AnalyticForceMatchesFiniteDifferenceUndamped) {
    // Central FD of E_pol for undamped mutual induction (isolates μ–μ force).
    const double r0 = 3.0;
    const double h = 1e-5;
    const std::vector<double> alpha = {0.0, 1.0, 1.5};

    auto at = [&](double r) { return run_pim(r, 0.0, 0.0, alpha).e_pol; };
    const double dEdr = (at(r0 + h) - at(r0 - h)) / (2.0 * h);
    auto out = run_pim(r0, 0.0, 0.0, alpha);
    EXPECT_NEAR(out.fx[0], dEdr, 5e-6);
    EXPECT_NEAR(out.fx[1], -dEdr, 5e-6);
    EXPECT_NEAR(out.fx[0] + out.fx[1], 0.0, 1e-10);
}

TEST(PimInducedDipole, AnalyticForceMatchesFiniteDifferenceThole) {
    const double r0 = 3.0;
    const double h = 1e-5;
    const std::vector<double> alpha = {0.0, 1.0, 1.5};
    const double thole = 0.39;

    auto at = [&](double r) { return run_pim(r, thole, 0.0, alpha).e_pol; };
    const double dEdr = (at(r0 + h) - at(r0 - h)) / (2.0 * h);
    auto out = run_pim(r0, thole, 0.0, alpha);
    EXPECT_NEAR(out.fx[0], dEdr, 5e-6);
    EXPECT_NEAR(out.fx[1], -dEdr, 5e-6);
    EXPECT_NEAR(out.fy[0], 0.0, 1e-10);
    EXPECT_NEAR(out.fz[0], 0.0, 1e-10);
    EXPECT_NEAR(out.fx[0] + out.fx[1], 0.0, 1e-10);
}

TEST(PimInducedDipole, CtorRejectsBadCutoff) {
    EXPECT_THROW(cpu::PimInducedDipole(-1.0, {1.0}), std::invalid_argument);
    EXPECT_THROW(cpu::PimInducedDipole(1.0, {-0.1}), std::invalid_argument);
}

TEST(PimInducedDipole, SccThrowsWhenUnconverged) {
    auto f = ion_pair(0.8, 1.0, -1.0, 1, 2);
    cpu::Context ctx;
    // Huge α + tight tol + max_iter=1 → throw
    cpu::PimInducedDipole pim(20.0, {0.0, 80.0, 80.0}, 0.0, 1e-16, 1);
    pim.build({ctx, f});
    EXPECT_THROW(pim.launch({ctx, f, 0, 0}), std::runtime_error);
}

TEST(PimInducedDipole, VirialDimerMatchesPairForm) {
    // Two atoms: W = r_01 ⊗ F_01, so W_xx = (x0−x1)·F0x for the axis-aligned
    // dimer; every other Voigt component vanishes (transverse forces are 0).
    const double r = 3.0;
    auto f = ion_pair(r, 1.0, -1.0, 1, 2);
    for (int c = 0; c < 6; ++c)
        f.meta.get(kStress)[c] = 0;
    cpu::Context ctx;
    cpu::PimInducedDipole pim(20.0, {0.0, 1.0, 1.5}, 0.39, 1e-12, 200);
    pim.build({ctx, f});
    pim.launch({ctx, f, 0, 0});

    const double f0x = f.span(kAtom_fx)[0];
    const double w_xx_want = (0.0 - r) * f0x; // r_01 = x0 − x1 = −r
    EXPECT_NEAR(f.meta.get(kStress)[0], w_xx_want, 1e-12);
    EXPECT_LT(f.meta.get(kStress)[0], 0.0); // attraction ⇒ tensile
    for (int c = 1; c < 6; ++c)
        EXPECT_NEAR(f.meta.get(kStress)[c], 0.0, 1e-12) << "Voigt c=" << c;
}

TEST(PimInducedDipole, VirialTraceMatchesFdScalingUndamped) {
    // Undamped forces are exact gradients, so trace(W) = −dE/ds under
    // uniform scaling of coordinates AND box (s = 1 ± h). Non-symmetric
    // 4-ion periodic configuration; min-image active (L/2 < spread paths).
    const double L0 = 8.0;
    const double x0[4] = {0.0, 2.5, 0.4, 6.9};
    const double y0[4] = {0.0, 0.6, 2.8, 6.2};
    const double z0[4] = {0.0, 0.3, 1.1, 7.0};
    const double q[4] = {1.0, -1.0, 1.0, -1.0};
    const std::vector<double> alpha = {0.0, 1.0, 1.5};
    const int t[4] = {1, 2, 1, 2};

    auto energy_and_stress = [&](double s, double* w6) {
        cpu::Frame f(4);
        for (int i = 0; i < 4; ++i) {
            f.span(kAtom_x)[i] = s * x0[i];
            f.span(kAtom_y)[i] = s * y0[i];
            f.span(kAtom_z)[i] = s * z0[i];
            f.span(kAtom_charge)[i] = q[i];
            f.span(kAtom_type)[i] = t[i];
            f.span(kAtom_mass)[i] = 1.0;
            f.span(kAtom_tag)[i] = i + 1;
            f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] =
                0;
        }
        f.box = cpu::Box::cubic(s * L0);
        f.meta.get(kEnergy) = 0;
        for (int c = 0; c < 6; ++c)
            f.meta.get(kStress)[c] = 0;
        cpu::Context ctx;
        cpu::PimInducedDipole pim(3.9, alpha, /*thole=*/0.0, 1e-13, 500);
        pim.build({ctx, f});
        pim.launch({ctx, f, 0, 0});
        if (w6)
            for (int c = 0; c < 6; ++c)
                w6[c] = f.meta.get(kStress)[c];
        return f.meta.get(kEnergy);
    };

    double w[6];
    energy_and_stress(1.0, w);
    const double trace = w[0] + w[1] + w[2];
    const double h = 1e-5;
    const double dEds =
        (energy_and_stress(1.0 + h, nullptr)
         - energy_and_stress(1.0 - h, nullptr))
        / (2.0 * h);
    ASSERT_NE(trace, 0.0);
    EXPECT_NEAR(trace, -dEds, std::max(1e-8, 1e-5 * std::fabs(trace)));
}

TEST(PimInducedDipole, VirialUntouchedWhenUnpolarizable) {
    auto f = ion_pair(3.0, 1.0, -1.0, 1, 2);
    for (int c = 0; c < 6; ++c)
        f.meta.get(kStress)[c] = 0;
    cpu::Context ctx;
    cpu::PimInducedDipole pim(20.0, {0.0, 0.0, 0.0}, 0.39, 1e-12, 200);
    pim.build({ctx, f});
    pim.launch({ctx, f, 0, 0});
    for (int c = 0; c < 6; ++c)
        EXPECT_EQ(f.meta.get(kStress)[c], 0.0) << "Voigt c=" << c;
}
