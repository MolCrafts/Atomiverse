/// @file tests/cpu/scf/test_fci.cpp
/// @brief Unit checks for RHF::fci(frame) (no public IntegralPack).

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame make_he() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[0] = 2;
    f.span(kAtom_mass)[0] = 4.0026;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

cpu::Frame make_h2() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.simbox = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

TEST(FCI, HeSto3gMatchesRHF) {
    auto frame = make_he();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    auto r = rhf.fci(frame);
    EXPECT_TRUE(rhf.converged());
    EXPECT_NEAR(r.e_corr, 0.0, 1e-12);
    EXPECT_NEAR(r.e_fci, rhf.energy(), 1e-12);
}

TEST(FCI, H2Sto3gLowersEnergy) {
    auto frame = make_h2();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    auto r = rhf.fci(frame);
    EXPECT_TRUE(rhf.converged());
    EXPECT_LE(r.e_corr, 1e-12); // correlation <= 0
    EXPECT_LE(r.e_fci, rhf.energy() + 1e-10);
}

#endif
