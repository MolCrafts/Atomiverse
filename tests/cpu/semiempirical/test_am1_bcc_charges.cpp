/// Constructor validation and molecular input errors, without an SCF solve.
#include "atomiverse/cpu/semiempirical/am1_bcc_charges.h"
#include "atomiverse/molpy.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using ATV::cpu::semiempirical::AM1;
using ATV::cpu::semiempirical::AM1BCCCharges;

TEST(AM1BCCCharges, CorrectionNamesAreValidated) {
    EXPECT_EQ(AM1BCCCharges{}.correction(), "bcc");
    EXPECT_EQ((AM1BCCCharges{AM1{}, "abcg2"}.correction()), "abcg2");
    EXPECT_THROW((AM1BCCCharges{AM1{}, "gaff2"}), std::invalid_argument);
}

TEST(AM1BCCCharges, MoleculeWithoutBondsIsRefused) {
    auto frame = molrs::frame_new();
    const std::vector<std::uint64_t> zs = {6, 6};
    molrs::frame_set_column_u64(*frame, "atoms", "atomic_number",
                                {zs.data(), zs.size()});
    auto mol = molrs::atomistic_from_frame(*frame);
    EXPECT_THROW(AM1BCCCharges{}.assign(*mol), std::invalid_argument);
}

TEST(AM1BCCCharges, MoleculeWithoutCoordinatesIsRefused) {
    auto frame = molrs::frame_new();
    const std::vector<std::uint64_t> zs = {1, 1}, i = {0}, j = {1};
    molrs::frame_set_column_u64(*frame, "atoms", "atomic_number",
                                {zs.data(), zs.size()});
    molrs::frame_set_column_u64(*frame, "bonds", "atomi", {i.data(), i.size()});
    molrs::frame_set_column_u64(*frame, "bonds", "atomj", {j.data(), j.size()});
    auto mol = molrs::atomistic_from_frame(*frame);
    EXPECT_THROW(AM1BCCCharges{}.assign(*mol), std::invalid_argument);
}
