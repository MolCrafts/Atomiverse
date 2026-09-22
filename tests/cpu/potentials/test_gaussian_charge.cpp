/// @file tests/cpu/potentials/test_gaussian_charge.cpp
/// Unit tests for GaussianChargeKernel + PairGaussianCoul (E2 electrode kernels).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/gaussian_charge.h"
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"

using namespace ATV;

namespace {

cpu::Frame dimer(double r, double q0, double q1, int t0, int t1) {
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
    f.box = cpu::Box::cubic(100.0);
    return f;
}

} // namespace

TEST(GaussianChargeKernel, EtaEffEqualWidths) {
    const double eta = 1.0;
    EXPECT_NEAR(cpu::GaussianChargeKernel::eta_eff(eta, eta),
                eta / std::sqrt(2.0), 1e-15);
}

TEST(GaussianChargeKernel, EtaEffPointPartner) {
    EXPECT_NEAR(cpu::GaussianChargeKernel::eta_eff(1.2, 0.0), 1.2, 1e-15);
    EXPECT_NEAR(cpu::GaussianChargeKernel::eta_eff(0.0, 0.8), 0.8, 1e-15);
}

TEST(GaussianChargeKernel, SelfGreens) {
    const double eta = 1.0;
    EXPECT_NEAR(cpu::GaussianChargeKernel::self_greens(eta),
                eta * std::sqrt(2.0 / ATV::units::pi), 1e-14);
}

TEST(GaussianChargeKernel, GreensPointLimit) {
    const double r = 3.5;
    EXPECT_NEAR(cpu::GaussianChargeKernel::greens(r, 0.0, 0.0), 1.0 / r, 1e-15);
    // Large η → erf→1 → 1/r
    EXPECT_NEAR(cpu::GaussianChargeKernel::greens(r, 50.0, 50.0), 1.0 / r,
                1e-12);
}

TEST(GaussianChargeKernel, DGreensMatchesFD) {
    const double r = 2.5, eta = 0.9;
    const double h = 1e-6;
    const double g_p = cpu::GaussianChargeKernel::greens(r + h, eta, eta);
    const double g_m = cpu::GaussianChargeKernel::greens(r - h, eta, eta);
    const double fd = (g_p - g_m) / (2.0 * h);
    EXPECT_NEAR(cpu::GaussianChargeKernel::dgreens_dr(r, eta, eta), fd, 1e-8);
}

TEST(PairGaussianCoul, AnalyticDimerEnergyForce) {
    const double r = 3.0, eta = 1.0, q0 = 1.0, q1 = 1.0;
    const double K = ATV::units::coulomb;
    const double G = cpu::GaussianChargeKernel::greens(r, eta, eta);
    const double dG = cpu::GaussianChargeKernel::dgreens_dr(r, eta, eta);
    const double U_pair = K * q0 * q1 * G;
    const double U_self = cpu::GaussianChargeKernel::self_energy(K, q0, eta)
                          + cpu::GaussianChargeKernel::self_energy(K, q1, eta);
    const double U = U_pair + U_self;
    // F_0x = (K q0 q1 dG/dr)  (r_vec along +x, F_i = (dU/dr) * r̂)
    const double F0x = K * q0 * q1 * dG;

    auto f = dimer(r, q0, q1, /*t0=*/1, /*t1=*/1);
    f.meta.get(kEnergy) = 0;
    for (int a = 0; a < 2; ++a)
        f.span(kAtom_fx)[a] = f.span(kAtom_fy)[a] = f.span(kAtom_fz)[a] = 0;

    cpu::Context ctx;
    cpu::PairGaussianCoul pot(20.0, /*eta_by_type=*/{0.0, eta}, true);
    pot.build({ctx, f});
    pot.launch({ctx, f, 0, 0});

    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[0], F0x, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[1], -F0x, 1e-12);
    pot.destroy({ctx, f});
}

TEST(PairGaussianCoul, PointPointRecoversCoulomb) {
    const double r = 4.0, q0 = 0.5, q1 = -1.0;
    const double K = ATV::units::coulomb;
    const double U = K * q0 * q1 / r;
    const double F0x = K * q0 * q1 * (-1.0 / (r * r)); // dG/dr = -1/r²

    auto f = dimer(r, q0, q1, 0, 0);
    f.meta.get(kEnergy) = 0;
    for (int a = 0; a < 2; ++a)
        f.span(kAtom_fx)[a] = f.span(kAtom_fy)[a] = f.span(kAtom_fz)[a] = 0;

    cpu::Context ctx;
    // empty eta table ⇒ all point
    cpu::PairGaussianCoul pot(20.0, {}, false);
    pot.build({ctx, f});
    pot.launch({ctx, f, 0, 0});

    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[0], F0x, 1e-12);
    pot.destroy({ctx, f});
}

TEST(PairGaussianCoul, GaussianPointCross) {
    const double r = 5.0, eta = 1.0, qg = 0.3, qp = 1.0;
    const double K = ATV::units::coulomb;
    const double G = cpu::GaussianChargeKernel::greens(r, eta, 0.0);
    const double U =
        K * qg * qp * G + cpu::GaussianChargeKernel::self_energy(K, qg, eta);

    auto f = dimer(r, qg, qp, /*gauss type*/ 1, /*point type*/ 0);
    f.meta.get(kEnergy) = 0;
    for (int a = 0; a < 2; ++a)
        f.span(kAtom_fx)[a] = f.span(kAtom_fy)[a] = f.span(kAtom_fz)[a] = 0;

    cpu::Context ctx;
    cpu::PairGaussianCoul pot(20.0, {0.0, eta}, true);
    pot.build({ctx, f});
    pot.launch({ctx, f, 0, 0});

    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    pot.destroy({ctx, f});
}

/// A type the width table does not cover is a missing parameter, not a point
/// charge. Returning 0 for it silently reclassified an electrode site — the
/// energy, the force and the induced charge all change, and nothing says a
/// row was left out of the table. House rule: lookup tables throw.
TEST(PairGaussianCoul, ThrowsOnATypeTheWidthTableDoesNotCover) {
    auto f =
        dimer(4.0, 1.0, -1.0, /*t0=*/0, /*t1=*/2); // type 2, table has 0..1
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    cpu::PairGaussianCoul pot(20.0, /*eta_by_type=*/{0.0, 1.0}, true);
    pot.build({ctx, f});
    EXPECT_THROW(pot.launch({ctx, f, 0, 0}), std::out_of_range);
    pot.destroy({ctx, f});
}

/// The documented "no table ⇒ every atom is a point charge" mode is NOT a
/// fallback — it is a stated configuration — and must keep working for any
/// type index.
TEST(PairGaussianCoul, EmptyTableIsAllPointChargesAtAnyTypeIndex) {
    const double r = 4.0, q0 = 0.5, q1 = -1.0;
    const double U = ATV::units::coulomb * q0 * q1 / r;
    auto f = dimer(r, q0, q1, /*t0=*/7, /*t1=*/9);
    f.meta.get(kEnergy) = 0;
    cpu::Context ctx;
    cpu::PairGaussianCoul pot(20.0, {}, false);
    pot.build({ctx, f});
    EXPECT_NO_THROW(pot.launch({ctx, f, 0, 0}));
    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    pot.destroy({ctx, f});
}
