/// @file tests/cpu/io/test_cpu_io_extxyz.cpp
/// @brief Tests for cpu::load_extxyz -- (ext)XYZ loading through molpy.
///
/// Small deterministic inline fixtures only (no external file dependency):
/// correct atom count / box from the Lattice= header / per-atom type = Z, and
/// a file lacking a Lattice= header throws a named exception -- no fallback
/// box.

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/extxyz.h"

using ATV::cpu::Frame;
using ATV::cpu::load_extxyz;

namespace {

// Write `content` to a unique temp file and return its path.
std::string write_temp(const std::string& tag, const std::string& content) {
    const std::string path = (std::filesystem::temp_directory_path()
                              / ("atv_extxyz_" + tag + ".xyz"))
                                 .string();
    std::ofstream out(path);
    out << content;
    out.close();
    return path;
}

} // namespace

// ── small deterministic inline fixture ─────────────────────────────────────
TEST(CpuIoExtxyz, InlineFixtureLoadsAtomsBoxAndZ) {
    // 3 atoms (O, H, H) in an orthorhombic box via a Lattice= header.
    const std::string xyz = "3\n"
                            "Lattice=\"5.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 7.0\" "
                            "Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
                            "O 0.0 0.0 0.0\n"
                            "H 0.9 0.0 0.0\n"
                            "H -0.3 0.85 0.0\n";
    const std::string path = write_temp("inline", xyz);

    Frame f = load_extxyz(path);
    std::remove(path.c_str());

    ASSERT_EQ(f.nlocal(), 3);
    EXPECT_EQ(f.span(ATV::kAtom_atomic_number)[0], 8); // O
    EXPECT_EQ(f.span(ATV::kAtom_atomic_number)[1], 1); // H
    EXPECT_EQ(f.span(ATV::kAtom_atomic_number)[2], 1); // H

    EXPECT_NEAR(f.box.length(0), 5.0, 1e-6);
    EXPECT_NEAR(f.box.length(1), 6.0, 1e-6);
    EXPECT_NEAR(f.box.length(2), 7.0, 1e-6);

    EXPECT_EQ(f.span(ATV::kAtom_x)[1], ATV::StorageReal(0.9));
    EXPECT_EQ(f.span(ATV::kAtom_y)[2], ATV::StorageReal(0.85));
}

// ── missing Lattice header throws, no fallback box ─────────────────
TEST(CpuIoExtxyz, MissingLatticeThrows) {
    const std::string xyz = "2\n"
                            "Properties=species:S:1:pos:R:3\n"
                            "O 0.0 0.0 0.0\n"
                            "H 0.9 0.0 0.0\n";
    const std::string path = write_temp("nolattice", xyz);

    EXPECT_THROW(load_extxyz(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(CpuIoExtxyz, LatticeVectorsBecomeBoxColumnsWithoutRotatingPositions) {
    const std::string xyz = "1\nLattice=\"4 0 0 1.5 5 0 0.2 0.7 6\" "
                            "Properties=species:S:1:pos:R:3\nH 1.33 1.78 2.4\n";
    const auto path = write_temp("tilted", xyz);
    const auto frame = load_extxyz(path);
    std::remove(path.c_str());
    const auto& h = frame.box.cell();
    EXPECT_DOUBLE_EQ(h(0, 1), 1.5);
    EXPECT_DOUBLE_EQ(h(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(h(0, 2), 0.2);
    EXPECT_DOUBLE_EQ(h(1, 2), 0.7);
    const auto frac =
        frame.box.inv_cell()
        * ATV::Vec3{frame.span(ATV::kAtom_x)[0], frame.span(ATV::kAtom_y)[0],
                    frame.span(ATV::kAtom_z)[0]};
    EXPECT_NEAR(frac[0], 0.2, 1e-12);
    EXPECT_NEAR(frac[1], 0.3, 1e-12);
    EXPECT_NEAR(frac[2], 0.4, 1e-12);
}
