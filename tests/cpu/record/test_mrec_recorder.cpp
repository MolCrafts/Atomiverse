/// @file tests/cpu/record/test_mrec_recorder.cpp
/// @brief Unit tests for `cpu::MrecRecorder` — the MD-phase `*.mrec` sink.
///
/// Three concerns, one per test: the constructor guards, the stride gate plus
/// the committed frame count after teardown, and the content of frame 0 read
/// back through the molrs bridge (`read_first_frame`). The recorder is driven
/// through a minimal deterministic LJ `cpu::md::MD` run because `launch()`
/// reads `ctx.step` and `ctx.runtime` — there is no smaller way to drive it —
/// but nothing here is a physics claim: frame 0 is compared against the same
/// fixture advanced one step *without* a recorder. Serial, sub-second, no
/// MPI, tmp stores only.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
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
#include "atomiverse/cpu/record/mrec_recorder.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// ── Deterministic LJ fixture (same as test_xyz_recorder.cpp) ─────────────

constexpr int kN = 10;           // atoms
constexpr double kEps = 0.0104;  // [eV]
constexpr double kSigma = 3.4;   // [Å]
constexpr double kCutoff = 6.0;  // [Å]
constexpr double kSkin = 0.3;    // [Å]
constexpr double kDt = 1.0;      // [fs]
constexpr double kSpacing = 3.8; // [Å]
constexpr double kBox = 30.0;    // [Å]

/// @brief Two rows of five argon atoms in a 30 Å cubic box (Z = 18).
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
    frame.box = ATV::cpu::Box::cubic(kBox);
    return frame;
}

/// @brief Mount the fixture's force recipe + integrator on `sim`.
void mount_lj(cpu::md::MD& sim) {
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
}

/// @brief Fresh tmp store path for one test, pre-cleaned.
std::string fresh_tmp_store(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    return p.string();
}

void remove_store(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

} // namespace

// ── 1. Constructor guards ────────────────────────────────────────────────

// The store suffix is the format's identity (molrs refuses `.zarr`), and a
// cadence below one step has no meaning; both are rejected loudly, before
// anything touches the filesystem.
TEST(MrecRecorderTest, ConstructorRejectsWrongSuffixAndZeroFreq) {
    EXPECT_THROW(cpu::MrecRecorder("traj.zarr", 1), std::invalid_argument);
    EXPECT_THROW(cpu::MrecRecorder("traj", 1), std::invalid_argument);
    EXPECT_THROW(cpu::MrecRecorder("", 1), std::invalid_argument);
    EXPECT_THROW(cpu::MrecRecorder("traj.mrec", 0), std::invalid_argument);
    EXPECT_NO_THROW(cpu::MrecRecorder("traj.mrec", 1));
}

// ── 2. Cadence + commit on teardown ──────────────────────────────────────

// Steps 0…5 at freq 2 open the gate on 0, 2, 4: three appends. Nothing is
// committed until the run is torn down (the driver's destructor fires
// `destroy()`), after which reopening the store through the bridge must
// report exactly three committed frames.
TEST(MrecRecorderTest, StrideGateAppendsThreeFramesCommittedAtTeardown) {
    const std::string path = fresh_tmp_store("atv_mrec_recorder_cadence.mrec");

    // Borrowed (`set_record_ref`) so the counters stay reachable. Declared
    // BEFORE the driver: the driver holds a non-owning reference and must be
    // destroyed first.
    cpu::MrecRecorder rec{path, /*freq=*/2};
    {
        cpu::Frame frame = make_lj_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        sim.run(frame, /*n_steps=*/6);
        EXPECT_EQ(rec.n_frames_written(), 3)
            << "steps 0…5 at freq 2 gate open on 0, 2 and 4 only";
    } // ~MD → flush_live_ → destroy() → close(): the buffered tail lands here

    ASSERT_TRUE(std::filesystem::is_directory(path));
    EXPECT_EQ(rec.n_frames_committed(), 3u)
        << "close() must commit every appended frame";

    // Independent check through the bridge: reattach and ask the store.
    auto reopened = molrs::trajectory_writer_open({path.data(), path.size()},
                                                  /*flush_every=*/1,
                                                  /*durable=*/false);
    EXPECT_EQ(molrs::trajectory_writer_committed(*reopened), 3u);
    molrs::trajectory_writer_close(std::move(reopened));

    remove_store(path);
}

// ── 3. Frame 0 content ───────────────────────────────────────────────────

// Frame 0 of a freq-1 run is the state at end_of_step of step 0, i.e. the
// fixture advanced by exactly one Velocity-Verlet step. Read it back through
// `read_first_frame` and compare against the same fixture advanced one step
// with no recorder attached — same deterministic recipe, no RNG anywhere.
TEST(MrecRecorderTest, FirstFrameReadsBackCoordinatesSpeciesAndBox) {
    const std::string path = fresh_tmp_store("atv_mrec_recorder_content.mrec");

    {
        cpu::Frame frame = make_lj_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record(cpu::MrecRecorder{path, /*freq=*/1});
        sim.build();
        sim.run(frame, /*n_steps=*/3);
    }

    cpu::Frame reference = make_lj_frame();
    {
        cpu::md::MD sim;
        mount_lj(sim);
        sim.build();
        sim.run(reference, /*n_steps=*/1);
    }

    auto fr = molrs::read_first_frame({path.data(), path.size()});
    auto x = molrs::frame_column_f64(*fr, "atoms", "x");
    auto y = molrs::frame_column_f64(*fr, "atoms", "y");
    auto z = molrs::frame_column_f64(*fr, "atoms", "z");
    auto species = molrs::frame_column_u32(*fr, "atoms", "atomic_number");
    auto mass = molrs::frame_column_f64(*fr, "atoms", "mass");
    auto box = molrs::frame_box(*fr);

    ASSERT_EQ(x.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(y.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(z.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(species.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(mass.size(), static_cast<std::size_t>(kN));
    const double* rx = reference.span(ATV::kAtom_x);
    const double* ry = reference.span(ATV::kAtom_y);
    const double* rz = reference.span(ATV::kAtom_z);
    for (int i = 0; i < kN; ++i) {
        SCOPED_TRACE("atom " + std::to_string(i));
        const auto k = static_cast<std::size_t>(i);
        EXPECT_TRUE(std::isfinite(x[k]) && std::isfinite(y[k])
                    && std::isfinite(z[k]));
        EXPECT_NEAR(x[k], rx[i], 1e-9);
        EXPECT_NEAR(y[k], ry[i], 1e-9);
        EXPECT_NEAR(z[k], rz[i], 1e-9);
        EXPECT_EQ(species[k], 18u) << "argon travels as atomic_number";
        EXPECT_DOUBLE_EQ(mass[k], 39.948);
    }
    ASSERT_EQ(box.size(), 9u);
    EXPECT_DOUBLE_EQ(box[0], kBox);
    EXPECT_DOUBLE_EQ(box[4], kBox);
    EXPECT_DOUBLE_EQ(box[8], kBox);
    EXPECT_DOUBLE_EQ(box[1], 0.0);

    remove_store(path);
}
