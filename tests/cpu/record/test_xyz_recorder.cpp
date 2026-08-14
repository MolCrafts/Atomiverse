/// @file tests/cpu/record/test_xyz_recorder.cpp
/// @brief Unit tests for `cpu::XyzRecorder` — the MD-phase XYZ trajectory sink.
///
/// Spec: .claude/specs/py-recorder-xyz (ac-003). Three concerns, one per test:
/// the stride gate (cadence), truncate-then-append across recorders, and the
/// content of a written frame. The recorder is exercised through a minimal
/// deterministic LJ `cpu::md::MD` run because `launch()` reads `ctx.step` and
/// `ctx.runtime` — there is no smaller way to drive it — but nothing here is a
/// physics claim: the frames are parsed back and only the recorder's own
/// contract is asserted. Serial, sub-second, no MPI, tmp files only.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/xyz_recorder.h"

using namespace ATV;

namespace {

// ── Deterministic LJ fixture ────────────────────────────────────────────
//
// Argon-like parameters, literal positions, NO InitVelocity — the run has no
// RNG at all, so every one of these tests writes byte-identical trajectories
// on every invocation.

constexpr int kN = 10;           // atoms
constexpr double kEps = 0.0104;  // [eV]
constexpr double kSigma = 3.4;   // [Å]
constexpr double kCutoff = 6.0;  // [Å]
constexpr double kSkin = 0.3;    // [Å]
constexpr double kDt = 1.0;      // [fs]
constexpr double kSpacing = 3.8; // [Å]

/// @brief Two rows of five argon atoms in a 30 Å cubic box (type 18 → "Ar").
cpu::Frame make_lj_frame() {
    cpu::Frame frame(kN);
    for (int i = 0; i < kN; ++i) {
        const int col = i % 5;
        const int row = i / 5;
        frame.span(ATV::kAtom_x)[i] = 8.0 + kSpacing * static_cast<double>(col);
        frame.span(ATV::kAtom_y)[i] =
            10.0 + kSpacing * static_cast<double>(row);
        frame.span(ATV::kAtom_z)[i] = 15.0;
        frame.span(ATV::kAtom_mass)[i] = 39.948;
        frame.span(ATV::kAtom_type)[i] = 18; // argon
    }
    frame.box = ATV::cpu::Box::cubic(30.0);
    return frame;
}

// ── XYZ parse-back ──────────────────────────────────────────────────────

/// @brief One parsed XYZ frame: `cpu::XYZ` emits the molrs extended-XYZ
///        layout — count line, comment line (`Lattice=… <meta…> Properties=…`),
///        then one `species x y z` line per atom.
struct XyzFrame {
    int n_atoms = 0;
    std::string comment;
    std::vector<std::string> species;
    std::vector<std::array<double, 3>> pos;
};

/// @brief Parse every frame of an XYZ trajectory, strictly.
///
/// Strict on purpose: a malformed count line, a truncated frame or an atom
/// line that does not read as `symbol x y z` throws rather than silently
/// yielding fewer frames — a recorder that half-writes must not read as a
/// recorder that wrote fewer frames.
std::vector<XyzFrame> parse_xyz(const std::string& path) {
    std::ifstream is(path);
    if (!is)
        throw std::runtime_error("parse_xyz: cannot open " + path);
    std::vector<XyzFrame> frames;
    std::string line;
    while (std::getline(is, line)) {
        XyzFrame f;
        {
            std::istringstream ss(line);
            if (!(ss >> f.n_atoms))
                throw std::runtime_error(
                    "parse_xyz: atom-count line did not parse as an int: '"
                    + line + "'");
        }
        if (!std::getline(is, f.comment))
            throw std::runtime_error(
                "parse_xyz: frame truncated before its comment line");
        f.species.reserve(static_cast<std::size_t>(f.n_atoms));
        f.pos.reserve(static_cast<std::size_t>(f.n_atoms));
        for (int i = 0; i < f.n_atoms; ++i) {
            if (!std::getline(is, line))
                throw std::runtime_error("parse_xyz: frame truncated at atom "
                                         + std::to_string(i));
            std::istringstream ss(line);
            std::string symbol;
            std::array<double, 3> r{};
            if (!(ss >> symbol >> r[0] >> r[1] >> r[2]))
                throw std::runtime_error(
                    "parse_xyz: atom line did not parse as 'symbol x y z': '"
                    + line + "'");
            f.species.push_back(symbol);
            f.pos.push_back(r);
        }
        frames.push_back(std::move(f));
    }
    return frames;
}

/// @brief Fresh tmp path for one test, pre-cleaned.
std::string fresh_tmp_path(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove(p, ec);
    return p.string();
}

/// @brief One deterministic LJ MD run whose only recorder is an owned
///        `XyzRecorder{path, freq}`.
void run_lj_md_with_recorder(const std::string& path, int freq, int n_steps) {
    cpu::Frame frame = make_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_record(cpu::XyzRecorder{path, freq});
    sim.build();
    sim.run(frame, n_steps);
}

} // namespace

// ── 1. Cadence ──────────────────────────────────────────────────────────

// Step indexing, pinned deliberately rather than derived in the assertion:
// `cpu::md::MD::run(frame, n)` dispatches `step_once(next_step_())` with a
// ZERO-BASED, cumulative counter (cpu/driver/driver.h `next_step_()`), so a
// 10-step run hands `ctx.step` the values 0…9 — never 10. The in-launch gate
// `ctx.step % freq_ != 0 → return` therefore fires at steps 0 and 5 only:
// **2 frames**, not `steps/freq + 1 = 3` (that arithmetic assumes an extra
// end-of-run write at step 10, which no driver emits).
//
// Empirically confirmed 2026-08-14 on this fixture with the house pattern
// XyzRecorder mirrors — `cpu::minimizer::TrajectoryWriter{path, 5}` (identical
// stride gate) bound on `md::Phase::end_of_step` over `run(frame, 10)` wrote
// exactly 2 frames.
TEST(XyzRecorderTest, StrideGateWritesStepsZeroAndFiveOfATenStepRun) {
    const std::string path = fresh_tmp_path("atv_xyz_recorder_cadence.xyz");

    // Borrowed (`set_record_ref`) so `n_frames_written()` is still reachable
    // after the run. Declared BEFORE the driver: the driver holds a
    // non-owning reference and must be destroyed first.
    cpu::XyzRecorder rec{path, /*freq=*/5};
    {
        cpu::Frame frame = make_lj_frame();
        cpu::md::MD sim;
        sim.on(ATV::md::Phase::force)
            .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
        sim.on(ATV::md::Phase::force)
            .add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
        sim.set_integrator(cpu::md::VelocityVerlet{kDt});
        sim.set_record_ref(rec);
        sim.build();
        sim.run(frame, /*n_steps=*/10);
    }

    ASSERT_TRUE(std::filesystem::exists(path));
    const auto frames = parse_xyz(path);
    EXPECT_EQ(frames.size(), 2u)
        << "steps 0…9 at freq 5 gate open on 0 and 5 only";
    EXPECT_EQ(rec.n_frames_written(), 2)
        << "the accessor must count the frames actually written";
    for (std::size_t i = 0; i < frames.size(); ++i) {
        SCOPED_TRACE("frame " + std::to_string(i));
        EXPECT_EQ(frames[i].n_atoms, kN);
        EXPECT_EQ(frames[i].pos.size(), static_cast<std::size_t>(kN));
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── 2. Truncate then append ─────────────────────────────────────────────

// The first launched write truncates, every later write of the SAME recorder
// appends. A second, fresh recorder on the same path therefore starts the
// file over — it must not extend the previous run's trajectory.
TEST(XyzRecorderTest, FreshRecorderOnSamePathTruncatesInsteadOfAppending) {
    const std::string path = fresh_tmp_path("atv_xyz_recorder_truncate.xyz");

    run_lj_md_with_recorder(path, /*freq=*/2, /*n_steps=*/6);
    ASSERT_TRUE(std::filesystem::exists(path));
    const auto first = parse_xyz(path);
    // steps 0…5 at freq 2 → 0, 2, 4.
    ASSERT_EQ(first.size(), 3u);

    run_lj_md_with_recorder(path, /*freq=*/2, /*n_steps=*/6);
    const auto second = parse_xyz(path);
    EXPECT_EQ(second.size(), first.size())
        << "a fresh recorder must truncate the file, not append 2x frames";

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── 3. Content sanity ───────────────────────────────────────────────────

// One frame, parsed back through the extended-XYZ layout `cpu::XYZ` writes:
// count line, comment line carrying the cell and the typed meta the recorder
// stamps, then `symbol x y z` per atom. The meta ORDER inside the comment is
// not asserted — molrs iterates a hash map, so only presence is stable.
TEST(XyzRecorderTest, WrittenFrameCarriesLatticeStepAndFiniteCoordinates) {
    const std::string path = fresh_tmp_path("atv_xyz_recorder_content.xyz");

    run_lj_md_with_recorder(path, /*freq=*/1, /*n_steps=*/1);

    ASSERT_TRUE(std::filesystem::exists(path));
    const auto frames = parse_xyz(path);
    ASSERT_EQ(frames.size(), 1u);
    const XyzFrame& f = frames.front();

    EXPECT_EQ(f.n_atoms, kN) << "count line must be the fixture atom count";
    EXPECT_NE(f.comment.find("Lattice="), std::string::npos) << f.comment;
    EXPECT_NE(f.comment.find("Properties="), std::string::npos) << f.comment;
    // The recorder stamps kStep into frame meta before writing, so the
    // iteration identity travels with the geometry.
    EXPECT_NE(f.comment.find("step="), std::string::npos) << f.comment;

    ASSERT_EQ(f.species.size(), static_cast<std::size_t>(kN));
    for (int i = 0; i < kN; ++i) {
        SCOPED_TRACE("atom " + std::to_string(i));
        EXPECT_EQ(f.species[static_cast<std::size_t>(i)], "Ar");
        for (int d = 0; d < 3; ++d)
            EXPECT_TRUE(std::isfinite(f.pos[static_cast<std::size_t>(i)][d]))
                << "coordinate " << d << " is not a finite number";
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}
