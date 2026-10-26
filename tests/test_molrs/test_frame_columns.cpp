/// @file tests/test_molrs/test_frame_columns.cpp
/// @brief Drift fence for molrs Frame column / box / meta FFI used by ATV.

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/molpy.hpp"
#include "rust/cxx.h"

namespace {

::rust::Box<molrs::FrameRef> one_atom_frame() {
    auto fr = molrs::frame_new();
    const std::vector<double> x = {1.0};
    const std::vector<double> y = {2.0};
    const std::vector<double> z = {3.0};
    const std::vector<std::uint64_t> znum = {8};
    const std::vector<double> mass = {15.999};
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
    const std::vector<double> h = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    molrs::frame_set_box(*fr, ::rust::Slice<const double>(h.data(), h.size()),
                         ::rust::Slice<const double>(
                             std::array<double, 3>{0.0, 0.0, 0.0}.data(), 3),
                         ::rust::Slice<const bool>(
                             std::array<bool, 3>{true, true, true}.data(), 3));
    return fr;
}

TEST(MolrsFrameColumns, FrameNewHasAtomsBlockAfterWrite) {
    auto fr = one_atom_frame();
    EXPECT_TRUE(molrs::frame_has_block(*fr, "atoms"));
    EXPECT_EQ(molrs::frame_block_n_rows(*fr, "atoms"), 1);
}

TEST(MolrsFrameColumns, RoundTripF64Columns) {
    auto fr = one_atom_frame();
    auto x = molrs::frame_column_f64(*fr, "atoms", "x");
    ASSERT_EQ(x.size(), 1u);
    EXPECT_DOUBLE_EQ(x[0], 1.0);
}

TEST(MolrsFrameColumns, RoundTripU32AtomicNumber) {
    auto fr = one_atom_frame();
    auto z = molrs::frame_column_u64(*fr, "atoms", "atomic_number");
    ASSERT_EQ(z.size(), 1u);
    EXPECT_EQ(z[0], 8u);
}

TEST(MolrsFrameColumns, RoundTripI32SpinColumn) {
    auto fr = molrs::frame_new();
    const std::vector<std::int32_t> spin = {1, -1};
    molrs::frame_set_column_i32(
        *fr, "atoms", "spin",
        ::rust::Slice<const std::int32_t>(spin.data(), spin.size()));
    auto got = molrs::frame_column_i32(*fr, "atoms", "spin");
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], 1);
    EXPECT_EQ(got[1], -1);
}

TEST(MolrsFrameColumns, RoundTripU32TypeIdOnBonds) {
    auto fr = molrs::frame_new();
    const std::vector<std::uint64_t> bi = {0};
    const std::vector<std::uint64_t> bj = {1};
    const std::vector<std::uint64_t> tid = {1};
    molrs::frame_set_column_u64(
        *fr, "bonds", "atomi",
        ::rust::Slice<const std::uint64_t>(bi.data(), bi.size()));
    molrs::frame_set_column_u64(
        *fr, "bonds", "atomj",
        ::rust::Slice<const std::uint64_t>(bj.data(), bj.size()));
    molrs::frame_set_column_u64(
        *fr, "bonds", "type_id",
        ::rust::Slice<const std::uint64_t>(tid.data(), tid.size()));
    auto got = molrs::frame_column_u64(*fr, "bonds", "type_id");
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0], 1u);
}

TEST(MolrsFrameColumns, MissingColumnOrBlockThrows) {
    auto fr = one_atom_frame();
    EXPECT_THROW(molrs::frame_column_f64(*fr, "atoms", "nope"), rust::Error);
    EXPECT_THROW(molrs::frame_column_f64(*fr, "missing", "x"), rust::Error);
}

// A column read as another dtype is the caller's schema bug: an error, never
// an empty read that looks like an absent column.
TEST(MolrsFrameColumns, MismatchedDtypeThrows) {
    auto fr = one_atom_frame();
    EXPECT_THROW(molrs::frame_column_i32(*fr, "atoms", "x"), rust::Error);
    EXPECT_THROW(molrs::frame_column_f64(*fr, "atoms", "atomic_number"),
                 rust::Error);
    EXPECT_THROW(molrs::frame_column_u64(*fr, "atoms", "mass"), rust::Error);
    EXPECT_THROW(molrs::frame_column_str(*fr, "atoms", "x"), rust::Error);
}

TEST(MolrsFrameColumns, BoxRoundTripNineDoubles) {
    auto fr = one_atom_frame();
    auto h = molrs::frame_box(*fr).h;
    ASSERT_EQ(h.size(), 9u);
    EXPECT_DOUBLE_EQ(h[0], 10.0);
    EXPECT_DOUBLE_EQ(h[4], 10.0);
    EXPECT_DOUBLE_EQ(h[8], 10.0);
}

TEST(MolrsFrameColumns, MetaF64RoundTrip) {
    auto fr = one_atom_frame();
    molrs::KeyedMetaValue entry{};
    entry.key = rust::String("energy");
    entry.dtype = molrs::MetaType::F64;
    entry.f64_value = -1.5;
    molrs::frame_set_meta(*fr, entry);
    auto keys = molrs::frame_meta_keys(*fr);
    ASSERT_EQ(keys.size(), 1u);
    EXPECT_EQ(std::string(keys[0]), "energy");
    const auto e = molrs::frame_get_meta(*fr, "energy");
    EXPECT_EQ(e.dtype, molrs::MetaType::F64);
    EXPECT_DOUBLE_EQ(e.f64_value, -1.5);
    EXPECT_THROW(molrs::frame_get_meta(*fr, "absent"), rust::Error);
}

TEST(MolrsFrameColumns, BlockNamesAndColumnsListAtoms) {
    auto fr = one_atom_frame();
    auto names = molrs::frame_block_names(*fr);
    bool has_atoms = false;
    for (const auto& n : names) {
        if (std::string(n) == "atoms")
            has_atoms = true;
    }
    EXPECT_TRUE(has_atoms);
    auto cols = molrs::frame_block_columns(*fr, "atoms");
    EXPECT_FALSE(cols.empty());
}

// frame_clone_from_addr is the Python capsule door — it takes the opaque
// Store address from molpy._ffi, not &FrameRef. Covered in python tests;
// here we only assert the symbol links by mentioning the type.
TEST(MolrsFrameColumns, FrameCloneFromAddrSymbolExists) {
    using Fn = decltype(&molrs::frame_clone_from_addr);
    static_assert(std::is_function_v<std::remove_pointer_t<Fn>>);
    EXPECT_NE(static_cast<Fn>(molrs::frame_clone_from_addr), nullptr);
}

} // namespace
