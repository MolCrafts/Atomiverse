/// @file tests/cpu/scf/test_uhf_solver.cpp
/// @brief UHF via the concrete member API (RHF/UHF, Frame in / out).

#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame make_h() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

cpu::Frame make_h2(double r = 0.74) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(UHF, HydrogenAtomConverges) {
    auto frame = make_h();
    cpu::scf::UHF uhf(1, 0);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.run(frame);
    using Spin = cpu::scf::SlaterDeterminant::Spin;
    EXPECT_TRUE(uhf.converged());
    EXPECT_EQ(uhf.determinant().n_electrons(Spin::alpha), 1);
    EXPECT_EQ(uhf.determinant().n_electrons(Spin::beta), 0);
    EXPECT_LT(uhf.energy(), 0.0);
}

TEST(UHF, H2SingletMatchesRHFShape) {
    auto frame = make_h2();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.run(frame);

    cpu::scf::UHF uhf(1, 1);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.run(frame);
    EXPECT_TRUE(rhf.converged());
    EXPECT_TRUE(uhf.converged());
    // Closed-shell UHF should be near RHF for H2 equilibrium
    EXPECT_NEAR(rhf.energy(), uhf.energy(), 1e-4);
}

TEST(UHF, ForcesWriteToFrame) {
    auto frame = make_h2();
    cpu::scf::UHF uhf(1, 1);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    auto g = uhf.forces(frame);
    EXPECT_EQ(g.forces.shape(0), 2u);
    EXPECT_TRUE(std::isfinite(g.energy));
}

#endif
