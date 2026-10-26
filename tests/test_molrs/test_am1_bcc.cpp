/// @file tests/test_molrs/test_am1_bcc.cpp
/// @brief Drift fence for molrs::BccModel, on the frame shape
///        `AM1BCCCharges` sends: atoms.{atomic_number, x, y, z} and
///        bonds.{atomi, atomj} — no `element`, no bond orders.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/molpy.hpp"

namespace {

::rust::Box<molrs::FrameRef> methane_like() {
    auto fr = molrs::frame_new();
    const std::vector<double> x = {0.0, 0.63, -0.63, -0.63, 0.63};
    const std::vector<double> y = {0.0, 0.63, -0.63, 0.63, -0.63};
    const std::vector<double> z = {0.0, 0.63, 0.63, -0.63, -0.63};
    const std::vector<std::uint64_t> znum = {6, 1, 1, 1, 1};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));
    molrs::frame_set_column_u64(
        *fr, "atoms", "atomic_number",
        ::rust::Slice<const std::uint64_t>(znum.data(), znum.size()));
    const std::vector<std::uint64_t> bi = {0, 0, 0, 0};
    const std::vector<std::uint64_t> bj = {1, 2, 3, 4};
    molrs::frame_set_column_u64(
        *fr, "bonds", "atomi",
        ::rust::Slice<const std::uint64_t>(bi.data(), bi.size()));
    molrs::frame_set_column_u64(
        *fr, "bonds", "atomj",
        ::rust::Slice<const std::uint64_t>(bj.data(), bj.size()));
    return fr;
}

// BCCPARM.DAT C(sp3)–H increment: +0.0393 e moved from C to each H.
constexpr double kCHIncrement = 0.0393;
constexpr double kExactTol = 1.0e-12;

TEST(MolrsAm1Bcc, CorrectAddsTheBondIncrements) {
    auto fr = methane_like();
    const std::vector<double> base = {-0.266, 0.066, 0.066, 0.066, 0.066};
    auto out = molrs::bcc_model_new({"bcc"})->correct(
        *fr, ::rust::Slice<const double>(base.data(), base.size()));
    ASSERT_EQ(out.size(), 5u);
    EXPECT_NEAR(out[0], -0.266 + 4.0 * kCHIncrement, kExactTol);
    for (std::size_t i = 1; i < 5; ++i)
        EXPECT_NEAR(out[i], 0.066 - kCHIncrement, kExactTol) << "i=" << i;
}

TEST(MolrsAm1Bcc, AssignEquivalencesBeforeCorrecting) {
    // The four hydrogens are one equivalence class: assign averages their
    // unequal base charges (mean 0.066) before adding the increments, so
    // the result equals the equivalenced correct() above.
    auto fr = methane_like();
    const std::vector<double> base = {-0.266, 0.056, 0.076, 0.066, 0.066};
    auto out = molrs::bcc_model_new({"bcc"})->assign(
        *fr, ::rust::Slice<const double>(base.data(), base.size()));
    ASSERT_EQ(out.size(), 5u);
    EXPECT_NEAR(out[0], -0.266 + 4.0 * kCHIncrement, kExactTol);
    for (std::size_t i = 1; i < 5; ++i)
        EXPECT_NEAR(out[i], 0.066 - kCHIncrement, kExactTol) << "i=" << i;
}

TEST(MolrsAm1Bcc, UnknownParameterSetErrors) {
    EXPECT_THROW(molrs::bcc_model_new({"not-a-set"}), std::exception);
}

TEST(MolrsAm1Bcc, WrongChargeCountErrors) {
    auto fr = methane_like();
    const std::vector<double> base = {0.0};
    EXPECT_THROW(
        molrs::bcc_model_new({"bcc"})->correct(
            *fr, ::rust::Slice<const double>(base.data(), base.size())),
        std::exception);
}

} // namespace
