/// @file tests/cpu/scf/test_uhf_ctor.cpp
/// @brief cpu::scf::UHF basis/occupation constructor assigns parameters and
///        does not run SCF.

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/uhf.h"

using ATV::cpu::scf::BasisSet;
using ATV::cpu::scf::UHF;

TEST(UhfCtor, AssignsBasisOccupationChargeAndTol) {
    const UHF uhf{BasisSet::STO3G, /*n_alpha=*/5, /*n_beta=*/4,
                  /*charge=*/1,    /*tol=*/1e-12, /*max_iter=*/50};
    EXPECT_EQ(uhf.basis, BasisSet::STO3G);
    EXPECT_EQ(uhf.n_alpha, 5);
    EXPECT_EQ(uhf.n_beta, 4);
    EXPECT_EQ(uhf.charge, 1);
    EXPECT_DOUBLE_EQ(uhf.tol, 1e-12);
    EXPECT_EQ(uhf.max_iter, 50);
    EXPECT_EQ(uhf.name(), "uhf");
}

TEST(UhfCtor, TrailingArgumentsDefault) {
    const UHF uhf{BasisSet::STO3G, /*n_alpha=*/1, /*n_beta=*/0};
    EXPECT_EQ(uhf.n_alpha, 1);
    EXPECT_EQ(uhf.n_beta, 0);
    EXPECT_EQ(uhf.charge, 0);
    EXPECT_DOUBLE_EQ(uhf.tol, 1e-8);
    EXPECT_EQ(uhf.max_iter, 100);
}

#endif
