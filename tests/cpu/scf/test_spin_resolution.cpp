/// @file tests/cpu/scf/test_spin_resolution.cpp
/// @brief SpinResolution + UHF/RHF occupation guards via concrete methods.

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/scf_types.h" // SpinResolution only
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.7572;
    f.span(kAtom_x)[2] = -0.7572;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_y)[2] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = -0.5860;
    f.span(kAtom_z)[2] = -0.5860;
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_mass)[2] = 1.008;
    f.simbox = cpu::Box::cubic(50.0);
    return f;
}

} // namespace

TEST(SpinResolution, ValidEvenElectronsMultiplicity1) {
    // H2O: 10 electrons, singlet
    cpu::scf::SpinResolution occ(10, 1);
    EXPECT_EQ(occ.n_alpha, 5);
    EXPECT_EQ(occ.n_beta, 5);
}

TEST(SpinResolution, ValidOddElectronsDoublet) {
    cpu::scf::SpinResolution occ(9, 2);
    EXPECT_EQ(occ.n_alpha, 5);
    EXPECT_EQ(occ.n_beta, 4);
}

TEST(SpinResolution, RejectsParityMismatch) {
    EXPECT_THROW(cpu::scf::SpinResolution(10, 2), std::invalid_argument);
}

TEST(UHF, OccupationMismatchThrows) {
    auto frame = make_water(); // 10 e-
    cpu::scf::UHF uhf(5, 4);   // sum 9
    uhf.basis = cpu::scf::BasisSet::STO3G;
    EXPECT_THROW(uhf.run(frame), std::invalid_argument);
}

TEST(RHF, OddElectronThrows) {
    auto frame = make_water();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.charge = 1; // 9 electrons
    EXPECT_THROW(rhf.run(frame), std::invalid_argument);
}

TEST(UHF, CorrectOccupationConverges) {
    auto frame = make_water();
    cpu::scf::UHF uhf(5, 5);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.run(frame);
    EXPECT_TRUE(uhf.converged());
}

#endif
