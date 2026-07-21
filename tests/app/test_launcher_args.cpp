/// @file tests/app/test_launcher_args.cpp
/// @brief Unit tests for the launcher's `ATV::app::LauncherArgs` — the
///        validate-in-the-constructor command line of `atomiverse <script.py>`
///        (mpi-launcher-embedded-python-04, ac-002 / ac-006).
///
/// `app/atomiverse/launcher_args.h` is header-only and NOT part of
/// libatomiverse: this test compiles it directly via a
/// `target_include_directories(... ${CMAKE_SOURCE_DIR}/app/atomiverse)` in
/// tests/CMakeLists.txt. Contract: construction succeeds iff argv carries a
/// readable script path; missing argument, nonexistent path and unreadable
/// file all throw std::runtime_error (mpirun replicates argv on every rank,
/// so the validation verdict is deterministic and identical across ranks).
/// Serial, millisecond-scale, temp files only.

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "launcher_args.h"

using ATV::app::LauncherArgs;

namespace {

/// @brief Build a mutable argv vector from string literals (LauncherArgs
///        takes `char**`, string literals are `const char*`).
class Argv {
  public:
    explicit Argv(std::vector<std::string> args) : storage_(std::move(args)) {
        for (auto& s : storage_)
            ptrs_.push_back(s.data());
        ptrs_.push_back(nullptr);
    }
    int argc() const { return static_cast<int>(storage_.size()); }
    char** argv() { return ptrs_.data(); }

  private:
    std::vector<std::string> storage_;
    std::vector<char*> ptrs_;
};

/// @brief RAII temp file under the gtest temp dir; removed on destruction.
class TempScript {
  public:
    explicit TempScript(const std::string& stem) {
        path_ = ::testing::TempDir() + stem + "."
                + std::to_string(static_cast<long long>(::getpid())) + ".py";
        std::ofstream out(path_);
        out << "print('hello from atv')\n";
    }
    ~TempScript() {
        ::chmod(path_.c_str(), 0644); // make removable even after chmod 000
        std::remove(path_.c_str());
    }
    const std::string& path() const { return path_; }

  private:
    std::string path_;
};

// ── Missing script argument ───────────────────────────────────────────────

TEST(LauncherArgs, MissingScriptArgumentThrows) {
    Argv args({"atomiverse"});
    EXPECT_THROW(LauncherArgs(args.argc(), args.argv()), std::runtime_error);
}

// ── Nonexistent path ──────────────────────────────────────────────────────

TEST(LauncherArgs, NonexistentScriptPathThrows) {
    Argv args({"atomiverse", "/no/such/dir/atv_missing_script.py"});
    EXPECT_THROW(LauncherArgs(args.argc(), args.argv()), std::runtime_error);
}

// ── Unreadable file ───────────────────────────────────────────────────────

TEST(LauncherArgs, UnreadableScriptThrows) {
    if (::geteuid() == 0)
        GTEST_SKIP() << "root ignores file permission bits; the unreadable "
                        "shape cannot be produced";

    TempScript script("atv_unreadable");
    ASSERT_EQ(::chmod(script.path().c_str(), 0000), 0);

    Argv args({"atomiverse", script.path()});
    EXPECT_THROW(LauncherArgs(args.argc(), args.argv()), std::runtime_error);
}

// ── Readable file: the happy path ─────────────────────────────────────────

TEST(LauncherArgs, ReadableScriptConstructsAndReturnsPath) {
    TempScript script("atv_readable");

    Argv args({"atomiverse", script.path()});
    const LauncherArgs parsed(args.argc(), args.argv());
    EXPECT_EQ(parsed.script(), script.path());
}

} // namespace
