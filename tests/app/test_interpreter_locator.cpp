/// @file tests/app/test_interpreter_locator.cpp
/// @brief Unit tests for `ATV::app::InterpreterLocator` — runtime discovery
///        of the Python interpreter the launcher embeds
///        (mpi-launcher-embedded-python-05, ac-004 / ac-005).
///
/// Contract under test (pure path/string logic, Python.h-free, no process
/// spawning — the injection constructor is the test seam):
///   1. a non-null ATV_PYTHON override wins, but must name an existing
///      executable file (access X_OK) — otherwise throw std::runtime_error
///      whose message names "ATV_PYTHON";
///   2. else scan PATH (':'-separated, empty segments skipped) left-to-right
///      for an executable `python3`, then a SECOND full pass for `python`
///      (so a python3 in a later dir beats a python in an earlier dir);
///   3. nothing found -> throw listing the search order + the ATV_PYTHON hint.
/// `check_abi` is a pure string compare: mismatch throws naming BOTH
/// versions and ATV_PYTHON.
///
/// `app/atomiverse/interpreter_locator.cpp` is NOT part of libatomiverse:
/// tests/CMakeLists.txt compiles that TU directly into this target and adds
/// `${CMAKE_SOURCE_DIR}/app/atomiverse` as an include dir (same treatment as
/// test_app_launcher_args). Fake interpreters are ofstream+chmod artifacts
/// under the gtest temp dir and are NEVER executed — these are
/// path-resolution tests only. Serial, millisecond-scale, zero third-party
/// reads.

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "interpreter_locator.h"

using ATV::app::InterpreterLocator;

namespace {

/// @brief RAII tree of fake-interpreter directories under the gtest temp
///        dir. `add_exe` drops a two-line shell stub with mode 0755 (or any
///        requested mode); nothing here is ever executed.
class FakePathTree : public ::testing::Test {
  protected:
    void SetUp() override {
        root_ = ::testing::TempDir() + "atv_locator."
                + std::to_string(static_cast<long long>(::getpid()));
        ASSERT_EQ(::mkdir(root_.c_str(), 0755), 0)
            << "cannot create fixture root " << root_;
        dirs_.push_back(root_);
    }

    void TearDown() override {
        for (auto it = files_.rbegin(); it != files_.rend(); ++it)
            std::remove(it->c_str());
        for (auto it = dirs_.rbegin(); it != dirs_.rend(); ++it)
            ::rmdir(it->c_str());
    }

    /// @brief Create (and track) a subdirectory of the fixture root.
    std::string add_dir(const std::string& name) {
        const std::string dir = root_ + "/" + name;
        EXPECT_EQ(::mkdir(dir.c_str(), 0755), 0);
        dirs_.push_back(dir);
        return dir;
    }

    /// @brief Create (and track) a fake interpreter file. Default mode 0755
    ///        = "executable"; pass 0644 for the not-executable shape.
    std::string add_exe(const std::string& dir, const std::string& name,
                        ::mode_t mode = 0755) {
        const std::string path = dir + "/" + name;
        {
            std::ofstream out(path);
            out << "#!/bin/sh\nexit 0\n"; // never run; access(X_OK) bait only
        }
        EXPECT_EQ(::chmod(path.c_str(), mode), 0);
        files_.push_back(path);
        return path;
    }

    /// @brief Construct and expect a std::runtime_error; return its message.
    static std::string throw_message(const char* override_value,
                                     const char* path_env) {
        try {
            InterpreterLocator locator(override_value, path_env);
            ADD_FAILURE() << "expected std::runtime_error, got executable()="
                          << locator.executable();
            return {};
        } catch (const std::runtime_error& err) {
            return err.what();
        }
    }

  private:
    std::string root_;
    std::vector<std::string> dirs_;
    std::vector<std::string> files_;
};

// ── (a) ATV_PYTHON override wins over PATH ────────────────────────────────

TEST_F(FakePathTree, OverrideWinsOverPathWithPython3) {
    const std::string path_dir = add_dir("bin_a");
    add_exe(path_dir, "python3");
    const std::string override_dir = add_dir("override");
    const std::string override_exe = add_exe(override_dir, "my_python");

    const InterpreterLocator locator(override_exe.c_str(), path_dir.c_str());
    EXPECT_EQ(locator.executable(), override_exe);
}

// ── (b) override pointing at a nonexistent file ───────────────────────────

TEST_F(FakePathTree, NonexistentOverrideThrowsNamingAtvPython) {
    const std::string missing = add_dir("empty") + "/no_such_python";

    const std::string msg = throw_message(missing.c_str(), nullptr);
    EXPECT_NE(msg.find("ATV_PYTHON"), std::string::npos) << msg;
}

// ── (c) override exists but is not executable ─────────────────────────────

TEST_F(FakePathTree, NonExecutableOverrideThrows) {
    const std::string dir = add_dir("noexec");
    const std::string plain_file = add_exe(dir, "python3", 0644);

    EXPECT_THROW(InterpreterLocator(plain_file.c_str(), nullptr),
                 std::runtime_error);
}

// ── (d) PATH search: left-to-right over ':'-separated dirs ────────────────

TEST_F(FakePathTree, Python3FoundInSecondPathDir) {
    const std::string first = add_dir("first"); // no interpreter here
    const std::string second = add_dir("second");
    const std::string expected = add_exe(second, "python3");
    const std::string path_env = first + ":" + second;

    const InterpreterLocator locator(nullptr, path_env.c_str());
    EXPECT_EQ(locator.executable(), expected);
}

TEST_F(FakePathTree, Python3InBothDirsFirstWins) {
    const std::string first = add_dir("first");
    const std::string second = add_dir("second");
    const std::string expected = add_exe(first, "python3");
    add_exe(second, "python3");
    const std::string path_env = first + ":" + second;

    const InterpreterLocator locator(nullptr, path_env.c_str());
    EXPECT_EQ(locator.executable(), expected);
}

// ── (e) `python` fallback is a SECOND full pass ───────────────────────────

TEST_F(FakePathTree, PlainPythonFoundWhenNoPython3Anywhere) {
    const std::string dir = add_dir("only_python");
    const std::string expected = add_exe(dir, "python");

    const InterpreterLocator locator(nullptr, dir.c_str());
    EXPECT_EQ(locator.executable(), expected);
}

TEST_F(FakePathTree, Python3InLaterDirBeatsPythonInEarlierDir) {
    const std::string early = add_dir("early");
    const std::string late = add_dir("late");
    add_exe(early, "python"); // earlier dir, but only the fallback name
    const std::string expected = add_exe(late, "python3");
    const std::string path_env = early + ":" + late;

    const InterpreterLocator locator(nullptr, path_env.c_str());
    EXPECT_EQ(locator.executable(), expected);
}

// ── (f) nothing found -> throw listing the search order + the hint ────────

TEST_F(FakePathTree, UnsetPathAndNoOverrideThrowsListingSearchOrder) {
    const std::string msg = throw_message(nullptr, nullptr);
    EXPECT_NE(msg.find("ATV_PYTHON"), std::string::npos) << msg;
    EXPECT_NE(msg.find("python3"), std::string::npos) << msg;
}

TEST_F(FakePathTree, EmptyPathAndNoOverrideThrowsNamingAtvPython) {
    const std::string msg = throw_message(nullptr, "");
    EXPECT_NE(msg.find("ATV_PYTHON"), std::string::npos) << msg;
}

TEST_F(FakePathTree, InterpreterFreePathDirsThrow) {
    const std::string dir = add_dir("barren"); // exists, holds no interpreter
    EXPECT_THROW(InterpreterLocator(nullptr, dir.c_str()), std::runtime_error);
}

// ── (g) empty PATH segments ("::", leading/trailing ':') are skipped ──────

TEST_F(FakePathTree, EmptyPathSegmentsAreSkippedWithoutCrash) {
    const std::string dir = add_dir("bin_g");
    const std::string expected = add_exe(dir, "python3");
    const std::string path_env = ":" + std::string("") + ":" + dir + "::";

    const InterpreterLocator locator(nullptr, path_env.c_str());
    EXPECT_EQ(locator.executable(), expected);
}

// ── (h) check_abi: pure version-string guard ──────────────────────────────

TEST(InterpreterLocatorAbi, MismatchThrowsNamingBothVersionsAndAtvPython) {
    try {
        InterpreterLocator::check_abi("3.11", "3.12");
        ADD_FAILURE() << "expected std::runtime_error on 3.11 vs 3.12";
    } catch (const std::runtime_error& err) {
        const std::string msg = err.what();
        EXPECT_NE(msg.find("3.11"), std::string::npos) << msg;
        EXPECT_NE(msg.find("3.12"), std::string::npos) << msg;
        EXPECT_NE(msg.find("ATV_PYTHON"), std::string::npos) << msg;
    }
}

TEST(InterpreterLocatorAbi, MatchingMinorVersionsDoNotThrow) {
    EXPECT_NO_THROW(InterpreterLocator::check_abi("3.12", "3.12"));
}

} // namespace
