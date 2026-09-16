/// @file tests/cpu/potentials/test_pair_buck.cpp
/// @brief Unit checks for cpu::PairBuck energy/force vs analytic shifted form.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_buck.h"

using namespace ATV;

namespace {

cpu::Frame dimer(double r) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);
    return f;
}

double U_un(double A, double rho, double C, double r) {
    return A * std::exp(-r / rho) - C / std::pow(r, 6);
}

double dUdr(double A, double rho, double C, double r) {
    return -A / rho * std::exp(-r / rho) + 6.0 * C / std::pow(r, 7);
}

} // namespace

TEST(PairBuck, AnalyticEnergyAndForce) {
    const double A = 1000.0, rho = 0.3, C = 50.0, rc = 8.0, r = 2.5;
    const double U = U_un(A, rho, C, r) - U_un(A, rho, C, rc);
    const double du = dUdr(A, rho, C, r);
    // F0x = dUdr (see PairYukawa / compute sign convention)

    auto f = dimer(r);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::PairBuck buck(A, rho, C, rc);
    buck.build({ctx, f});
    buck.launch({ctx, f, 0, 0});

    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[0], du, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[1], -du, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fy)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[0], 0.0, 1e-15);
    buck.destroy({ctx, f});
}

TEST(PairBuck, ZeroAtCutoff) {
    const double A = 1000.0, rho = 0.3, C = 50.0, rc = 8.0;
    // Place pair just inside cutoff — energy should approach 0 from above/below
    const double r = rc - 1e-6;
    const double U = U_un(A, rho, C, r) - U_un(A, rho, C, rc);

    auto f = dimer(r);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::PairBuck buck(A, rho, C, rc);
    buck.build({ctx, f});
    buck.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_LT(std::fabs(f.meta.get(kEnergy)), 1e-6);
    buck.destroy({ctx, f});
}

TEST(PairBuck, RejectsInvalidParams) {
    EXPECT_THROW(cpu::PairBuck(1.0, 0.0, 1.0, 5.0), std::invalid_argument);
    EXPECT_THROW(cpu::PairBuck(1.0, 0.3, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(cpu::PairBuck(-1.0, 0.3, 1.0, 5.0), std::invalid_argument);
    EXPECT_THROW(cpu::PairBuck(1.0, 0.3, -1.0, 5.0), std::invalid_argument);
}

TEST(PairBuck, NameIsBuck) {
    cpu::PairBuck buck(1.0, 0.3, 1.0, 5.0);
    EXPECT_EQ(buck.name(), "buck");
    EXPECT_EQ(buck.as_neighbor_consumer(), nullptr);
}
