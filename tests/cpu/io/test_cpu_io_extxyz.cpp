/// @file tests/cpu/io/test_cpu_io_extxyz.cpp
/// @brief Tests for cpu::load_extxyz -- (ext)XYZ loading through molrs.
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
    EXPECT_EQ(f.span(ATV::kAtom_type)[0], 8); // O
    EXPECT_EQ(f.span(ATV::kAtom_type)[1], 1); // H
    EXPECT_EQ(f.span(ATV::kAtom_type)[2], 1); // H

    EXPECT_NEAR(f.simbox.length(0), 5.0, 1e-6);
    EXPECT_NEAR(f.simbox.length(1), 6.0, 1e-6);
    EXPECT_NEAR(f.simbox.length(2), 7.0, 1e-6);

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
