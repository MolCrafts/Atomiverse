/// @file tests/cpu/task/test_launcher_gate.cpp
/// @brief Unit tests for `cpu::task::LauncherGate` — the "am I hosted by the
///        MPI launcher?" probe (mpi-launcher-embedded-python-04, ac-002).
///
/// Contract under test (spec revision 2026-07-15): `active()` ⇔ marker
/// `ATV_LAUNCHER` present ∧ marker PID == getpid() ∧ MPI initialized ∧
/// world size > 1 — and **every** inconsistency degrades to inactive,
/// never a throw. The child-process shape (inherited marker, foreign PID)
/// is SILENT; the own-PID-but-no-MPI shape warns on stderr exactly once
/// per process (std::once_flag). This binary never calls MPI_Init, so MPI
/// is uninitialized in every test here and `active()` must always be
/// false. Serial, millisecond-scale, no mpirun.

#include <gtest/gtest.h>

#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>

#include "atomiverse/cpu/task/launcher_gate.h"

using ATV::cpu::task::LauncherGate;

namespace {

// The never-throws contract is compile-time-pinnable: construction and the
// probe are both noexcept.
static_assert(std::is_nothrow_default_constructible_v<LauncherGate>,
              "LauncherGate() must never throw (spec: inconsistency "
              "degrades to inactive)");
static_assert(noexcept(std::declval<const LauncherGate&>().active()),
              "LauncherGate::active() must be noexcept");

/// @brief Save/restore the ATV_LAUNCHER marker around each test so the
///        suite leaves the process environment exactly as it found it.
class LauncherGateTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const char* prev = std::getenv("ATV_LAUNCHER");
        had_prev_ = (prev != nullptr);
        if (had_prev_)
            prev_ = prev;
        ::unsetenv("ATV_LAUNCHER");
    }

    void TearDown() override {
        if (had_prev_) {
            ::setenv("ATV_LAUNCHER", prev_.c_str(), /*overwrite=*/1);
        } else {
            ::unsetenv("ATV_LAUNCHER");
        }
    }

    /// @brief Construct a gate while capturing stderr; returns the pair
    ///        (active, captured stderr). active() is polled inside the
    ///        capture window so a warning emitted from either the ctor or
    ///        the probe is caught.
    static std::pair<bool, std::string> probe_with_stderr() {
        ::testing::internal::CaptureStderr();
        bool is_active = true;
        EXPECT_NO_THROW({
            const LauncherGate gate;
            is_active = gate.active();
        });
        return {is_active, ::testing::internal::GetCapturedStderr()};
    }

  private:
    bool had_prev_ = false;
    std::string prev_;
};

// ── No marker: plain `python atv.py` shape ────────────────────────────────

TEST_F(LauncherGateTest, NoMarkerIsInactiveAndSilent) {
    const auto [active, err] = probe_with_stderr();
    EXPECT_FALSE(active) << "no ATV_LAUNCHER marker must mean inactive";
    EXPECT_TRUE(err.empty())
        << "the serial path must be zero-noise, got stderr: " << err;
}

// ── Foreign-PID marker: the inherited-env child-process shape ─────────────
//
// A child python spawned from the launcher-hosted user script inherits
// ATV_LAUNCHER but has a new PID. This was the review defect the spec
// revision fixes: it must be SILENTLY inactive — no throw, no warning
// (a multiprocessing pool must not print one warning per worker).

TEST_F(LauncherGateTest, ForeignPidMarkerIsInactiveSilentAndDoesNotThrow) {
    const std::string foreign =
        std::to_string(static_cast<long long>(::getpid()) + 1);
    ASSERT_EQ(::setenv("ATV_LAUNCHER", foreign.c_str(), 1), 0);

    const auto [active, err] = probe_with_stderr();
    EXPECT_FALSE(active)
        << "a marker written by another process must not activate the gate";
    EXPECT_TRUE(err.empty())
        << "the child-process shape must be silent, got stderr: " << err;
}

// ── Unparseable marker: degrade, never throw ──────────────────────────────

TEST_F(LauncherGateTest, GarbageMarkerIsInactiveSilentAndDoesNotThrow) {
    ASSERT_EQ(::setenv("ATV_LAUNCHER", "not-a-pid", 1), 0);

    const auto [active, err] = probe_with_stderr();
    EXPECT_FALSE(active) << "an unparseable marker cannot match getpid()";
    EXPECT_TRUE(err.empty())
        << "a non-matching marker is the foreign shape: silent; got: " << err;
}

// ── Own-PID marker with MPI uninitialized: warn exactly once ──────────────
//
// This is the genuinely inconsistent host state (cannot occur in the real
// launcher flow, where PythonHost sets the marker only after MPI_Init).
// Both constructions stay inactive and nothrow; the stderr warning fires
// on the FIRST construction only (std::once_flag). Both constructions
// live in one test so the exactly-once claim cannot be broken by test
// ordering.

TEST_F(LauncherGateTest, OwnPidMarkerNoMpiWarnsExactlyOnceAndStaysInactive) {
    const std::string own = std::to_string(static_cast<long long>(::getpid()));
    ASSERT_EQ(::setenv("ATV_LAUNCHER", own.c_str(), 1), 0);

    const auto [active_first, err_first] = probe_with_stderr();
    EXPECT_FALSE(active_first)
        << "own-PID marker without an initialized MPI world must degrade "
           "to inactive";
    EXPECT_FALSE(err_first.empty())
        << "the inconsistent-host shape must emit a one-time stderr "
           "warning for diagnosability";

    const auto [active_second, err_second] = probe_with_stderr();
    EXPECT_FALSE(active_second);
    EXPECT_TRUE(err_second.empty())
        << "the warning is once per process (std::once_flag); second "
           "construction emitted: "
        << err_second;
}

} // namespace
