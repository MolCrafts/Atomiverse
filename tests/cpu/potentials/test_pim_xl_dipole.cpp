/// @file tests/cpu/potentials/test_pim_xl_dipole.cpp
/// @brief Unit tests for cpu::PimXlDipole (iEL/SCF extended-Lagrangian
///        dipoles) and the PimInducedDipole::mu_guess primitive.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"
#include "atomiverse/cpu/potentials/pim_xl_dipole.h"

using namespace ATV;

namespace {

/// Two-site ± toy of the pim-induced-dipole-bulk regression: q=+1 α=1.0 at
/// the origin, q=−1 α=1.5 at (r,0,0), L=50 (min-image ≡ free space).
cpu::Frame ion_pair(double r) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 2;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
    f.span(kAtom_fx)[1] = f.span(kAtom_fy)[1] = f.span(kAtom_fz)[1] = 0;
    f.box = cpu::Box::cubic(50.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

void zero_forces(cpu::Frame& f) {
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(kAtom_fx)[i] = 0;
        f.span(kAtom_fy)[i] = 0;
        f.span(kAtom_fz)[i] = 0;
    }
    f.meta.get(kEnergy) = 0;
}

const std::vector<double> kAlpha{0.0, 1.0, 1.5};

} // namespace

TEST(PimXlDipole, CtorRejectsBadXlOrder) {
    EXPECT_THROW(cpu::PimXlDipole(20.0, kAlpha, 0.39, /*xl_order=*/2),
                 std::invalid_argument);
    EXPECT_THROW(cpu::PimXlDipole(20.0, kAlpha, 0.39, /*xl_order=*/10),
                 std::invalid_argument);
}

TEST(PimXlDipole, CtorDelegatesPimValidation) {
    EXPECT_THROW(cpu::PimXlDipole(-1.0, kAlpha), std::invalid_argument);
    EXPECT_THROW(cpu::PimXlDipole(20.0, {0.0, -1.0}), std::invalid_argument);
}

TEST(PimXlDipole, NameAndPropagatorOrder) {
    cpu::PimXlDipole xl(20.0, kAlpha, 0.39, /*xl_order=*/7);
    EXPECT_EQ(xl.name(), "pim_xl_dipole");
    EXPECT_EQ(xl.propagator().order(), 7);
    EXPECT_DOUBLE_EQ(xl.propagator().kappa(),
                     cpu::electronic::XlBomd(7).kappa());
}

TEST(PimXlDipole, ColdLaunchMatchesPlainPim) {
    auto fa = ion_pair(3.0);
    auto fb = ion_pair(3.0);
    cpu::Context ctx;

    cpu::PimInducedDipole pim(20.0, kAlpha, 0.39, 1e-12, 200);
    pim.build({ctx, fa});
    pim.launch({ctx, fa, 0, 0});

    cpu::PimXlDipole xl(20.0, kAlpha, 0.39, 5, 1e-12, 200);
    xl.build({ctx, fb});
    xl.launch({ctx, fb, 0, 0});

    EXPECT_FALSE(xl.last_warm());
    EXPECT_TRUE(xl.last_pol_converged());
    EXPECT_DOUBLE_EQ(xl.last_pol_energy(), pim.last_pol_energy());
    EXPECT_EQ(xl.last_pol_iters(), pim.last_pol_iters());
    ASSERT_EQ(xl.last_mu().size(), pim.last_mu().size());
    for (std::size_t k = 0; k < pim.last_mu().size(); ++k)
        EXPECT_DOUBLE_EQ(xl.last_mu()[k], pim.last_mu()[k]);
    for (int i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(fb.span(kAtom_fx)[i], fa.span(kAtom_fx)[i]);
        EXPECT_DOUBLE_EQ(fb.span(kAtom_fy)[i], fa.span(kAtom_fy)[i]);
        EXPECT_DOUBLE_EQ(fb.span(kAtom_fz)[i], fa.span(kAtom_fz)[i]);
    }
}

TEST(PimXlDipole, WarmStartConvergesFasterAtFixedGeometry) {
    auto f = ion_pair(3.0);
    cpu::Context ctx;
    cpu::PimXlDipole xl(20.0, kAlpha, 0.39, 5, 1e-12, 200);
    xl.build({ctx, f});

    xl.launch({ctx, f, 0, 0});
    const int cold_iters = xl.last_pol_iters();
    const double e_cold = xl.last_pol_energy();
    EXPECT_FALSE(xl.last_warm());

    zero_forces(f);
    xl.launch({ctx, f, 0, 1});
    EXPECT_TRUE(xl.last_warm());
    EXPECT_LT(xl.last_pol_iters(), cold_iters);
    EXPECT_LE(xl.last_pol_iters(), 2);
    // Stationary geometry: the auxiliary fixed point IS the SCC solution.
    EXPECT_NEAR(xl.last_pol_energy(), e_cold, 1e-12);
    EXPECT_LT(xl.last_aux_error(), 1e-10);
}

TEST(PimXlDipole, MuGuessInvarianceOnPlainPim) {
    auto fa = ion_pair(3.0);
    auto fb = ion_pair(3.0);
    cpu::Context ctx;

    cpu::PimInducedDipole cold(20.0, kAlpha, 0.39, 1e-12, 200);
    cold.build({ctx, fa});
    cold.launch({ctx, fa, 0, 0});

    cpu::PimInducedDipole warm(20.0, kAlpha, 0.39, 1e-12, 200);
    warm.build({ctx, fb});
    warm.mu_guess = cold.last_mu(); // exact solution as guess
    warm.launch({ctx, fb, 0, 0});

    EXPECT_TRUE(warm.mu_guess.empty()) << "guess must be consumed";
    EXPECT_LE(warm.last_pol_iters(), 2);
    EXPECT_NEAR(warm.last_pol_energy(), cold.last_pol_energy(), 1e-12);
    ASSERT_EQ(warm.last_mu().size(), cold.last_mu().size());
    for (std::size_t k = 0; k < cold.last_mu().size(); ++k)
        EXPECT_NEAR(warm.last_mu()[k], cold.last_mu()[k], 1e-12);
}

TEST(PimXlDipole, WrongSizeGuessIsIgnored) {
    auto fa = ion_pair(3.0);
    auto fb = ion_pair(3.0);
    cpu::Context ctx;

    cpu::PimInducedDipole a(20.0, kAlpha, 0.39, 1e-12, 200);
    a.build({ctx, fa});
    a.launch({ctx, fa, 0, 0});

    cpu::PimInducedDipole b(20.0, kAlpha, 0.39, 1e-12, 200);
    b.build({ctx, fb});
    b.mu_guess = {1.0, 2.0}; // wrong size → cold start
    b.launch({ctx, fb, 0, 0});

    EXPECT_EQ(b.last_pol_iters(), a.last_pol_iters());
    EXPECT_DOUBLE_EQ(b.last_pol_energy(), a.last_pol_energy());
}

TEST(PimXlDipole, AutoResetOnAtomCountChange) {
    auto f2 = ion_pair(3.0);
    cpu::Context ctx;
    cpu::PimXlDipole xl(20.0, kAlpha, 0.39, 5, 1e-12, 200);
    xl.build({ctx, f2});
    xl.launch({ctx, f2, 0, 0});
    zero_forces(f2);
    xl.launch({ctx, f2, 0, 1});
    EXPECT_TRUE(xl.last_warm());

    cpu::Frame g(3); // 3-atom frame: the 2-site pair plus one more cation
    const double gx[3] = {0.0, 3.0, 6.0};
    const double gq[3] = {1.0, -1.0, 1.0};
    const int gt[3] = {1, 2, 1};
    for (int i = 0; i < 3; ++i) {
        g.span(kAtom_x)[i] = gx[i];
        g.span(kAtom_y)[i] = 0;
        g.span(kAtom_z)[i] = 0;
        g.span(kAtom_charge)[i] = gq[i];
        g.span(kAtom_type)[i] = gt[i];
        g.span(kAtom_mass)[i] = 1.0;
        g.span(kAtom_tag)[i] = i + 1;
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    }
    g.box = cpu::Box::cubic(50.0);
    g.meta.get(kEnergy) = 0;

    xl.launch({ctx, g, 0, 2}); // must not throw; cold restart
    EXPECT_FALSE(xl.last_warm());
    EXPECT_EQ(xl.last_mu().size(), 9u);
}

TEST(PimXlDipole, ResetDropsAuxHistory) {
    auto f = ion_pair(3.0);
    cpu::Context ctx;
    cpu::PimXlDipole xl(20.0, kAlpha, 0.39, 5, 1e-12, 200);
    xl.build({ctx, f});
    xl.launch({ctx, f, 0, 0});
    zero_forces(f);
    xl.launch({ctx, f, 0, 1});
    EXPECT_TRUE(xl.last_warm());

    xl.reset();
    EXPECT_FALSE(xl.propagator().ready());
    zero_forces(f);
    xl.launch({ctx, f, 0, 2});
    EXPECT_FALSE(xl.last_warm());
    EXPECT_TRUE(xl.last_pol_converged());
}
