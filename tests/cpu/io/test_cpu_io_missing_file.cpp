/// @file tests/cpu/io/test_cpu_io_missing_file.cpp
/// @brief Unreadable input paths throw std::runtime_error, never abort.
///
/// The molrs CXX bridge declares its readers without `Result`, so a Rust-side
/// I/O error is a panic and cxx aborts the process — a user `try/catch` would
/// never see it. Each loader must reject an unreadable path on the C++ side
/// first. One case per loader: mistyped (nonexistent) path, and a directory
/// handed to a file loader.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/extxyz.h"
#include "atomiverse/cpu/io/lammps_data.h"
#include "atomiverse/cpu/record/xyz.h"

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

// ── ZarrReader ─────────────────────────────────────────────────────────────
TEST(CpuIoMissingFile, ZarrReaderMissingStoreThrows) {
    const std::string path = missing_path("store.zarr");
    ASSERT_FALSE(std::filesystem::exists(path));

    ATV::cpu::Frame frame(1);
    ATV::cpu::ZarrReader reader(path);
    try {
        reader.read_into(frame);
        FAIL() << "ZarrReader::read_into did not throw on a nonexistent store";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("cannot read"), std::string::npos) << msg;
        EXPECT_NE(msg.find(path), std::string::npos) << msg;
    }
}

// ── Readable-but-corrupt input ─────────────────────────────────────────────
//
// This is the case molrs 0.13 changed. Before it, `xyz_read_first_frame` was
// declared without `Result`, so a parse failure was a Rust `panic!` and cxx
// aborted the process — no C++ handler could run and the test binary would
// die here. Since 0.13 it returns `Result`, cxx raises `rust::Error`, and
// `load_extxyz` rethrows it as a `std::runtime_error` naming the path.
TEST(CpuIoCorruptFile, LoadExtxyzCorruptFileThrowsWithPath) {
    const auto path =
        std::filesystem::temp_directory_path()
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
