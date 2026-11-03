/// @file tests/cpu/record/test_xyz_recorder.cpp
/// @brief Unit tests for `cpu::XyzRecorder` — the MD-phase XYZ trajectory sink.
///
/// Spec: .claude/specs/py-recorder-xyz (ac-003). One concern per test: the
/// stride gate (cadence), truncate-then-append across recorders, the content
/// of a written frame, and the LAMMPS step convention (frame `s` is the state
/// after `s` steps — checked against an independent driver that ran exactly
/// `s` steps — steps 0, k, …, n, a continued run writing its boundary step
/// once). The recorder is exercised through a minimal deterministic LJ
/// `cpu::md::MD` run because its hooks read `ctx.step` and `ctx.runtime` —
/// there is no smaller way to drive it — but nothing here is a physics claim.
/// Serial, sub-second, no MPI, tmp files only.

#include <gtest/gtest.h>

#include <algorithm>
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
        frame.span(ATV::kAtom_atomic_number)[i] = 18; // argon
    }
    frame.box = ATV::cpu::Box::cubic(30.0);
    return frame;
}

// ── XYZ parse-back ──────────────────────────────────────────────────────

/// @brief One parsed XYZ frame: `cpu::XYZ` emits the molpy extended-XYZ
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

// LAMMPS `dump` cadence (src/output.cpp Output::setup / Output::write): the
// run's setup writes the initial state (step 0) and the end of every step
// whose completed-step count is a multiple of freq writes another, so a
// 10-step run at freq 5 writes steps 0, 5 and 10 — **3 frames**, the final
// state included.
TEST(XyzRecorderTest, StrideGateWritesStepsZeroFiveAndTenOfATenStepRun) {
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
    EXPECT_EQ(frames.size(), 3u)
        << "a 10-step run at freq 5 writes steps 0, 5 and 10";
    EXPECT_EQ(rec.n_frames_written(), 3)
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
    // a 6-step run at freq 2 → steps 0, 2, 4, 6.
    ASSERT_EQ(first.size(), 4u);

    run_lj_md_with_recorder(path, /*freq=*/2, /*n_steps=*/6);
    const auto second = parse_xyz(path);
    EXPECT_EQ(second.size(), first.size())
        << "a fresh recorder must truncate the file, not append 2x frames";

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── 3. Content sanity ───────────────────────────────────────────────────

// The step-1 frame, parsed back through the extended-XYZ layout `cpu::XYZ` writes:
// count line, comment line carrying the cell and the typed meta the recorder
// stamps, then `symbol x y z` per atom. The meta ORDER inside the comment is
// not asserted — molpy iterates a hash map, so only presence is stable.
TEST(XyzRecorderTest, WrittenFrameCarriesLatticeStepAndFiniteCoordinates) {
    const std::string path = fresh_tmp_path("atv_xyz_recorder_content.xyz");

    run_lj_md_with_recorder(path, /*freq=*/1, /*n_steps=*/1);

    ASSERT_TRUE(std::filesystem::exists(path));
    const auto frames = parse_xyz(path);
    ASSERT_EQ(frames.size(), 2u) << "the initial state and step 1";
    const XyzFrame& f = frames.back();

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

// ── 4. LAMMPS step convention ───────────────────────────────────────────
//
// LAMMPS increments `update->ntimestep` at the top of every step
// (src/verlet.cpp Verlet::run), dumps at the end of a step whose timestep is
// a multiple of `every` (src/output.cpp Output::write), dumps the current
// state at the start of every run (Output::setup) and never writes a
// timestep twice (`last_dump`). The oracle is not the recorder: each frame
// is compared with an independent driver that ran exactly that many steps.

namespace {

/// @brief The fixture compressed to 3.5 Å with a deterministic velocity
///        field: every step moves every atom by ~1e-3 Å.
cpu::Frame make_moving_lj_frame() {
    cpu::Frame frame(kN);
    for (int i = 0; i < kN; ++i) {
        frame.span(ATV::kAtom_x)[i] = 8.0 + 3.5 * static_cast<double>(i % 5);
        frame.span(ATV::kAtom_y)[i] = 10.0 + 3.5 * static_cast<double>(i / 5);
        frame.span(ATV::kAtom_z)[i] = 15.0;
        frame.span(ATV::kAtom_vx)[i] = 0.002 * static_cast<double>(i % 3 - 1);
        frame.span(ATV::kAtom_vy)[i] = 0.001 * static_cast<double>(i % 2);
        frame.span(ATV::kAtom_vz)[i] = 0.0005 * static_cast<double>(i % 4 - 2);
        frame.span(ATV::kAtom_mass)[i] = 39.948;
        frame.span(ATV::kAtom_atomic_number)[i] = 18;
    }
    frame.box = ATV::cpu::Box::cubic(30.0);
    return frame;
}

void mount_lj(cpu::md::MD& sim) {
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
}

/// @brief Oracle: the moving fixture after exactly `steps` steps of a fresh
///        driver with no recorder.
cpu::Frame state_after(int steps) {
    cpu::Frame frame = make_moving_lj_frame();
    cpu::md::MD sim;
    mount_lj(sim);
    sim.build();
    sim.run(frame, steps);
    return frame;
}

/// @brief The `step=N` value of an extended-XYZ comment line.
long long comment_step(const std::string& comment) {
    const auto at = comment.find("step=");
    if (at == std::string::npos)
        throw std::runtime_error("no step= in '" + comment + "'");
    return std::stoll(comment.substr(at + 5));
}

/// @brief Every written atom sits on a distinct oracle atom (positions are
///        rows in no particular order, so match by nearest position).
void expect_frame_is(const XyzFrame& f, const cpu::Frame& ref) {
    ASSERT_EQ(f.n_atoms, kN);
    std::vector<bool> used(kN, false);
    for (int i = 0; i < kN; ++i) {
        const auto& p = f.pos[static_cast<std::size_t>(i)];
        int best = -1;
        double best_d = 1e300;
        for (int j = 0; j < kN; ++j) {
            const double d = std::abs(p[0] - ref.span(ATV::kAtom_x)[j])
                             + std::abs(p[1] - ref.span(ATV::kAtom_y)[j])
                             + std::abs(p[2] - ref.span(ATV::kAtom_z)[j]);
            if (d < best_d) {
                best_d = d;
                best = j;
            }
        }
        ASSERT_GE(best, 0);
        EXPECT_FALSE(used[static_cast<std::size_t>(best)]) << "atom " << i;
        used[static_cast<std::size_t>(best)] = true;
        EXPECT_LT(best_d, 1e-9) << "atom " << i << " is not the oracle state";
    }
}

} // namespace

// A fresh 6-step run at freq 3 writes steps 0, 3, 6, each frame the state
// after exactly that many steps; the last is the state run() handed back.
TEST(XyzRecorderTest,
     FreshRunWritesZeroThroughNEachTheStateAfterThatManySteps) {
    const std::string path = fresh_tmp_path("atv_xyz_recorder_lammps.xyz");
    cpu::Frame frame = make_moving_lj_frame();
    {
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record(cpu::XyzRecorder{path, /*freq=*/3});
        sim.build();
        sim.run(frame, /*n_steps=*/6);
    }
    const auto frames = parse_xyz(path);
    ASSERT_EQ(frames.size(), 3u);
    const long long want[] = {0, 3, 6};
    // The oracle separates consecutive steps by far more than 1e-9 Å.
    {
        const cpu::Frame a = state_after(3), b = state_after(4);
        double d = 0.0;
        for (int j = 0; j < kN; ++j)
            d = std::max(
                d, std::abs(a.span(ATV::kAtom_x)[j] - b.span(ATV::kAtom_x)[j]));
        ASSERT_GT(d, 1e-5);
    }
    for (std::size_t f = 0; f < frames.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        EXPECT_EQ(comment_step(frames[f].comment), want[f]);
        expect_frame_is(frames[f], state_after(static_cast<int>(want[f])));
    }
    expect_frame_is(frames.back(), frame);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Continued `run(cpu::Frame&, …)` calls extend one trajectory and write the
// boundary step once.
TEST(XyzRecorderTest, ContinuedRunWritesTheBoundaryStepOnce) {
    const std::string path = fresh_tmp_path("atv_xyz_recorder_continued.xyz");
    {
        cpu::Frame frame = make_moving_lj_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record(cpu::XyzRecorder{path, /*freq=*/2});
        sim.build();
        sim.run(frame, 4); // 0, 2, 4
        sim.run(frame, 4); // (4 already written), 6, 8
    }
    const auto frames = parse_xyz(path);
    ASSERT_EQ(frames.size(), 5u);
    for (std::size_t f = 0; f < frames.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        const long long step = comment_step(frames[f].comment);
        EXPECT_EQ(step, static_cast<long long>(2 * f));
        expect_frame_is(frames[f], state_after(static_cast<int>(step)));
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}
