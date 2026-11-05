/// @file tests/test_molpy/test_xyz_io.cpp
/// @brief Drift fence for molpy XYZ / single-frame IO doors ATV uses.

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/io/extxyz.h"
#include "atomiverse/cpu/io/frame_writer.h"
#include "atomiverse/molpy.hpp"
#include "rust/cxx.h"

namespace {

std::string tmp_path(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

void remove_path(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// The engine's exchange frame for a writer: atoms.{atomic_number, x, y, z},
// plus the cell when `box` is not empty.
::rust::Box<molrs::FrameRef> atoms_frame(const std::vector<std::uint64_t>& type,
                                         const std::vector<double>& x,
                                         const std::vector<double>& y,
                                         const std::vector<double>& z,
                                         const std::vector<double>& box) {
    auto fr = molrs::frame_new();
    molrs::frame_set_column_u64(
        *fr, "atoms", "atomic_number",
        ::rust::Slice<const std::uint64_t>(type.data(), type.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));
    if (!box.empty())
        molrs::frame_set_box(
            *fr, ::rust::Slice<const double>(box.data(), box.size()),
            ::rust::Slice<const double>(
                std::array<double, 3>{0.0, 0.0, 0.0}.data(), 3),
            ::rust::Slice<const bool>(
                std::array<bool, 3>{true, true, true}.data(), 3));
    return fr;
}

TEST(MolpyXyzIo, WriteXyzFrameThenReadXyzFrame) {
    const std::string path = tmp_path("atv_molpy_xyz_io.xyz");
    remove_path(path);
    const auto frame =
        atoms_frame({1, 1}, {0.0, 0.74}, {0.0, 0.0}, {0.0, 0.0}, {});
    molrs::write_xyz({path.data(), path.size()}, *frame, /*append=*/false);
    // The writer fills `element` on its own copy, never on the caller's frame.
    for (const rust::String& col : molrs::frame_block_columns(*frame, "atoms"))
        EXPECT_NE(std::string(col), "element");

    auto fr = molrs::read_xyz({path.data(), path.size()});
    auto xr = molrs::frame_column_f64(*fr, "atoms", "x");
    ASSERT_EQ(xr.size(), 2u);
    EXPECT_NEAR(xr[1], 0.74, 1e-6);
    // read_xyz converts the symbols explicitly and hands back Z only.
    auto zr = molrs::frame_column_u64(*fr, "atoms", "atomic_number");
    ASSERT_EQ(zr.size(), 2u);
    EXPECT_EQ(zr[0], 1u);
    EXPECT_EQ(zr[1], 1u);
    for (const rust::String& col : molrs::frame_block_columns(*fr, "atoms"))
        EXPECT_NE(std::string(col), "element");
    remove_path(path);
}

TEST(MolpyXyzIo, WriteXyzRefusesAFrameItCannotSpell) {
    const std::string path = tmp_path("atv_molpy_xyz_typed.xyz");
    remove_path(path);
    const std::vector<double> box = {5, 0, 0, 0, 5, 0, 0, 0, 5};
    molrs::write_xyz({path.data(), path.size()},
                     *atoms_frame({8}, {0.0}, {0.0}, {0.0}, box),
                     /*append=*/false);
    EXPECT_TRUE(std::filesystem::exists(path));
    remove_path(path);

    // Z = 119 has no symbol, and a frame without atomic_number has no species:
    // errors, never a placeholder symbol.
    EXPECT_THROW(molrs::write_xyz({path.data(), path.size()},
                                  *atoms_frame({119}, {0.0}, {0.0}, {0.0}, box),
                                  /*append=*/false),
                 rust::Error);
    auto no_species = molrs::frame_new();
    const std::vector<double> x = {0.0};
    molrs::frame_set_column_f64(
        *no_species, "atoms", "x",
        ::rust::Slice<const double>(x.data(), x.size()));
    EXPECT_THROW(molrs::write_xyz({path.data(), path.size()}, *no_species,
                                  /*append=*/false),
                 rust::Error);
    EXPECT_FALSE(std::filesystem::exists(path));

    // One atomic number for two atoms, and a 4-value H: the exchange frame
    // refuses them, never a silently truncated frame or one without its box.
    EXPECT_THROW(atoms_frame({8}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {}),
                 rust::Error);
    EXPECT_THROW(atoms_frame({8}, {0.0}, {0.0}, {0.0}, {5, 0, 0, 0}),
                 rust::Error);
}

TEST(MolpyXyzIo, CpuWriteXyzFrameWrapper) {
    const std::string path = tmp_path("atv_molpy_cpu_write_xyz.xyz");
    remove_path(path);
    const std::array<double, 9> box = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    ATV::cpu::write_xyz_frame(path, {1}, {0.0}, {0.0}, {0.0}, box,
                              /*append=*/false);
    EXPECT_TRUE(std::filesystem::exists(path));
    remove_path(path);

    // kAtom_type carries Z: a negative value has no atomic-number spelling
    // and is refused before it crosses; Z > 118 is molpy' refusal.
    EXPECT_THROW(ATV::cpu::write_xyz_frame(path, {-1}, {0.0}, {0.0}, {0.0}, box,
                                           /*append=*/false),
                 std::invalid_argument);
    EXPECT_THROW(ATV::cpu::write_xyz_frame(path, {119}, {0.0}, {0.0}, {0.0},
                                           box, /*append=*/false),
                 std::runtime_error);
    remove_path(path);
}

TEST(MolpyXyzIo, ExtxyzMissingFileThrows) {
    EXPECT_THROW(ATV::cpu::load_extxyz("/no/such/atv_molpy_missing.xyz"),
                 std::runtime_error);
}

} // namespace
