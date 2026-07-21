/// @file tests/cpu/scf/test_scf_type_homes.cpp
/// @brief scf-slater-determinant-02-rehome: public SCF vocabulary types live
///        in their own headers — enums in scf_enums.h, GradientResult in
///        gradient_result.h, SCFInitialGuess in scf_guess.h. This test
///        deliberately does NOT include scf_result.h: each type must be
///        reachable from its new home alone.

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/scf_enums.h"
#include "atomiverse/cpu/scf/scf_guess.h"

using namespace ATV;

TEST(ScfTypeHomes, MeanFieldAndSpinEnumsUsableFromScfEnumsHeader) {
    cpu::scf::MeanFieldType mf = cpu::scf::MeanFieldType::HF;
    EXPECT_NE(mf, cpu::scf::MeanFieldType::KS);

    cpu::scf::SpinType spin = cpu::scf::SpinType::Restricted;
    EXPECT_NE(spin, cpu::scf::SpinType::Unrestricted);
}

TEST(ScfTypeHomes, InitGuessEnumUsableFromScfEnumsHeader) {
    cpu::scf::InitGuess guess = cpu::scf::InitGuess::Hcore;
    EXPECT_NE(guess, cpu::scf::InitGuess::MINAO);
}

TEST(ScfTypeHomes, SCFInitialGuessConstructibleFromScfGuessHeader) {
    cpu::scf::SCFInitialGuess guess;
    EXPECT_FALSE(guess.has_alpha);
    EXPECT_FALSE(guess.has_beta);

    guess.P_alpha = xt::xtensor<double, 2>::from_shape({2, 2});
    guess.P_alpha.fill(0.5);
    guess.has_alpha = true;
    EXPECT_TRUE(guess.has_alpha);
    EXPECT_DOUBLE_EQ(guess.P_alpha(1, 1), 0.5);
}

TEST(ScfTypeHomes, GradientResultConstructibleFromGradientResultHeader) {
    cpu::scf::GradientResult grad;
    EXPECT_DOUBLE_EQ(grad.energy, 0.0);

    grad.forces = xt::xtensor<double, 2>::from_shape({3, 3});
    grad.forces.fill(0.0);
    grad.forces(2, 1) = -1.25; // eV/Å
    grad.energy = -76.0;       // eV
    EXPECT_DOUBLE_EQ(grad.forces(2, 1), -1.25);
    EXPECT_DOUBLE_EQ(grad.energy, -76.0);
}

#endif
