/// @file tests/test_molrs/test_frame_exchange.cpp
/// @brief Drift fence for cpu::from_molrs / cpu::to_molrs (ATV's Frame boundary).

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/molrs.hpp"

namespace {

::rust::Box<molrs::FrameRef> make_h2_molrs() {
    auto fr = molrs::frame_new();
    const std::vector<double> x = {0.0, 0.74};
    const std::vector<double> y = {0.0, 0.0};
    const std::vector<double> z = {0.0, 0.0};
    const std::vector<std::uint64_t> znum = {1, 1};
    const std::vector<double> mass = {1.008, 1.008};
    const std::vector<std::int32_t> tag = {1, 2};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));
    molrs::frame_set_column_u64(
        *fr, "atoms", "atomic_number",
        ::rust::Slice<const std::uint64_t>(znum.data(), znum.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "mass",
        ::rust::Slice<const double>(mass.data(), mass.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "tag",
        ::rust::Slice<const std::int32_t>(tag.data(), tag.size()));
    const std::vector<double> h = {20, 0, 0, 0, 20, 0, 0, 0, 20};
    molrs::frame_set_box_h(*fr,
                           ::rust::Slice<const double>(h.data(), h.size()));
    return fr;
}

TEST(MolrsFrameExchange, FromMolrsReadsPositionsAndNlocal) {
    auto fr = make_h2_molrs();
    ATV::cpu::Frame frame = ATV::cpu::from_molrs(*fr);
    EXPECT_EQ(frame.nlocal(), 2);
    EXPECT_NEAR(frame.span(ATV::kAtom_x)[0], 0.0, 1e-12);
    EXPECT_NEAR(frame.span(ATV::kAtom_x)[1], 0.74, 1e-12);
}

TEST(MolrsFrameExchange, ToMolrsWritesBackPositions) {
    auto fr = make_h2_molrs();
    ATV::cpu::Frame frame = ATV::cpu::from_molrs(*fr);
    frame.span(ATV::kAtom_x)[0] = 0.1;
    ATV::cpu::to_molrs(frame, *fr);
    auto x = molrs::frame_column_f64(*fr, "atoms", "x");
    ASSERT_EQ(x.size(), 2u);
    EXPECT_NEAR(x[0], 0.1, 1e-12);
}

TEST(MolrsFrameExchange, MissingAtomsBlockThrows) {
    auto fr = molrs::frame_new();
    EXPECT_THROW(ATV::cpu::from_molrs(*fr), std::runtime_error);
}

// Species cross as Z only. A frame spelling symbols alone is refused with the
// conversion named; after the explicit molrs conversion
// (frame_fill_atomic_numbers) it materializes with Z in kAtom_type.
TEST(MolrsFrameExchange, ElementSymbolsNeedTheExplicitConversion) {
    auto fr = molrs::frame_new();
    const std::vector<double> x = {0.0, 0.96};
    const std::vector<double> zero = {0.0, 0.0};
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y",
        ::rust::Slice<const double>(zero.data(), zero.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z",
        ::rust::Slice<const double>(zero.data(), zero.size()));
    const std::vector<::rust::String> symbols = {"O", "H"};
    molrs::frame_set_column_str(
        *fr, "atoms", "element",
        ::rust::Slice<const ::rust::String>(symbols.data(), symbols.size()));

    try {
        (void)ATV::cpu::from_molrs(*fr);
        FAIL() << "from_molrs accepted a frame without atomic_number";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("fill_atomic_numbers"),
                  std::string::npos)
            << e.what();
    }

    molrs::frame_fill_atomic_numbers(*fr);
    ATV::cpu::Frame frame = ATV::cpu::from_molrs(*fr);
    ASSERT_EQ(frame.nlocal(), 2);
    EXPECT_EQ(frame.span(ATV::kAtom_type)[0], 8);
    EXPECT_EQ(frame.span(ATV::kAtom_type)[1], 1);
}

} // namespace

TEST(MolrsFrameExchange, NonSymmetricCellKeepsColumnLatticeAcrossBoundary) {
    auto fr = make_h2_molrs();
    const std::vector<double> h = {4, 1.5, 0.2, 0, 5, 0.7, 0, 0, 6};
    molrs::frame_set_box_h(*fr,
                           ::rust::Slice<const double>(h.data(), h.size()));
    auto frame = ATV::cpu::from_molrs(*fr);
    const ATV::Vec3 frac{0.2, 0.3, 0.4};
    const auto cart = frame.box.cell() * frac;
    EXPECT_NEAR(cart[0], 1.33, 1e-12);
    EXPECT_NEAR(cart[1], 1.78, 1e-12);
    EXPECT_NEAR(cart[2], 2.4, 1e-12);
    frame.box.scale_isotropic(1.1);
    ATV::cpu::to_molrs(frame, *fr);
    const auto back = molrs::frame_box_h(*fr);
    ASSERT_EQ(back.size(), 9u);
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(back[i], 1.1 * h[i], 1e-12);
}
