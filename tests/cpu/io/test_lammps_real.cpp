/// @file tests/cpu/io/test_lammps_real.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/io/lammps_real.h"

using namespace ATV::cpu;

TEST(LammpsReal, KcalToEvFactor) {
    EXPECT_NEAR(kcal_per_mol_to_ev(1.0), 1.0 / 23.060547830619026, 1e-15);
    EXPECT_NEAR(kKcalPerMolInEv * 23.060547830619026, 1.0, 1e-15);
}

TEST(LammpsReal, DegreeToRad) {
    EXPECT_NEAR(degree_to_rad(109.5), 109.5 * M_PI / 180.0, 1e-15);
}

TEST(LammpsReal, FourierToCharmm) {
    FourierTermReal t{0.25, 1, 0.0};
    auto c = fourier_term_to_charmm(t);
    EXPECT_NEAR(c.K_ev, kcal_per_mol_to_ev(0.25), 1e-15);
    EXPECT_EQ(c.n, 1);
    EXPECT_NEAR(c.delta_rad, 0.0, 1e-15);
}

TEST(LammpsReal, MultiTermRejected) {
    EXPECT_THROW(reject_multi_term_fourier(2), std::invalid_argument);
    EXPECT_NO_THROW(reject_multi_term_fourier(1));
}

TEST(LammpsReal, BadMultiplicity) {
    FourierTermReal t{0.1, 0, 0.0};
    EXPECT_THROW(fourier_term_to_charmm(t), std::invalid_argument);
}
