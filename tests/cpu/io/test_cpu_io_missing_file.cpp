/// @file tests/cpu/io/test_cpu_io_missing_file.cpp
/// @brief Unreadable input paths throw std::runtime_error, never abort.
///
/// Each loader rejects an unreadable path on the C++ side first, so the
/// message names the path in Atomiverse's own wording; a failure inside a
/// readable file crosses from molpy as `rust::Error` and is rethrown naming
/// the path. One case per loader: mistyped (nonexistent) path, and a
/// directory handed to a file loader.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/extxyz.h"
#include "atomiverse/cpu/io/lammps_data.h"

#include <unistd.h>

namespace {

// A path under the temp dir that is guaranteed not to exist.
std::string missing_path(const char* tag) {
    return (std::filesystem::temp_directory_path()
            / (std::string("atv_no_such_file_") + tag))
        .string();
}

} // namespace

// ── load_extxyz ────────────────────────────────────────────────────────────
TEST(CpuIoMissingFile, LoadExtxyzMissingPathThrows) {
    const std::string path = missing_path("extxyz.xyz");
    ASSERT_FALSE(std::filesystem::exists(path));

    try {
        ATV::cpu::load_extxyz(path);
        FAIL() << "load_extxyz did not throw on a nonexistent path";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("cpu::load_extxyz"), std::string::npos) << msg;
        EXPECT_NE(msg.find("cannot read"), std::string::npos) << msg;
        EXPECT_NE(msg.find(path), std::string::npos) << msg;
    }
}

TEST(CpuIoMissingFile, LoadExtxyzDirectoryThrows) {
    const std::string dir = std::filesystem::temp_directory_path().string();
    EXPECT_THROW(ATV::cpu::load_extxyz(dir), std::runtime_error);
}

// ── load_lammps_data ───────────────────────────────────────────────────────
TEST(CpuIoMissingFile, LoadLammpsDataMissingPathThrows) {
    const std::string path = missing_path("lammps.data");
    ASSERT_FALSE(std::filesystem::exists(path));
    EXPECT_THROW(ATV::cpu::load_lammps_data(path), std::runtime_error);
}

// ── Readable-but-corrupt input ─────────────────────────────────────────────
//
// molpy `read_xyz` returns `Result`, so a parse failure is a
// `rust::Error` (not a Rust `panic!`, which cxx would turn into an abort),
// and `load_extxyz` rethrows it as a `std::runtime_error` naming the path.
TEST(CpuIoCorruptFile, LoadExtxyzCorruptFileThrowsWithPath) {
    const auto path = std::filesystem::temp_directory_path()
                      / ("atv_corrupt_" + std::to_string(::getpid()) + ".xyz");
    {
        std::ofstream f(path);
        f << "not-a-number\ngarbage garbage\n@@@\n";
    }
    ASSERT_TRUE(std::filesystem::exists(path));

    try {
        (void)ATV::cpu::load_extxyz(path.string());
        std::filesystem::remove(path);
        FAIL() << "load_extxyz did not throw on a corrupt file";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        std::filesystem::remove(path);
        // The wrapper must name the operation and the file; the bridge
        // message alone does neither.
        EXPECT_NE(msg.find("cpu::load_extxyz"), std::string::npos) << msg;
        EXPECT_NE(msg.find(path.string()), std::string::npos) << msg;
    }
}
