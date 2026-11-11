/// @file tests/cpu/record/test_mrec_recorder.cpp
/// @brief Unit tests for `cpu::MrecRecorder` — the MD-phase `*.mrec` sink.
///
/// One concern per test: the constructor guards, the stride gate plus the
/// committed frame count after teardown, the content of a frame read back
/// through the molpy bridge (`read_mrec_trajectory_frame`), row order,
/// identity columns, the status section, and the LAMMPS step convention
/// (frame `s` is the state after `s` steps; steps 0, k, …, n; time = s·dt;
/// a continued run writes its boundary step once). The recorder is driven
/// through a minimal deterministic LJ `cpu::md::MD` run because its hooks
/// read `ctx.step` and `ctx.runtime` — there is no smaller way to drive it —
/// but nothing here is a physics claim: every recorded state is compared
/// against the same fixture advanced exactly that many steps by an
/// independent driver *without* a recorder. Serial, sub-second, no MPI, tmp
/// stores only.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/mrec.h"
#include "atomiverse/cpu/record/mrec_recorder.h"
#include "atomiverse/molpy.hpp"

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
        frame.span(ATV::kAtom_atomic_number)[i] = 18; // argon
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

// The store suffix is the format's identity (molpy refuses `.zarr`), and a
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

// A 6-step run at freq 2 writes the initial state and the ends of steps 2,
// 4 and 6: four appends. Nothing is committed until the run is torn down
// (the driver's destructor fires `destroy()`), after which reopening the
// store through the bridge must report exactly four committed frames.
TEST(MrecRecorderTest, StrideGateAppendsFourFramesCommittedAtTeardown) {
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
        EXPECT_EQ(rec.n_frames_written(), 4)
            << "a 6-step run at freq 2 writes steps 0, 2, 4 and 6";
    } // ~MD → flush_live_ → destroy() → close(): the buffered tail lands here

    ASSERT_TRUE(std::filesystem::is_directory(path));
    EXPECT_EQ(rec.n_frames_committed(), 4u)
        << "close() must commit every appended frame";

    // Independent check through the bridge: reattach and ask the store.
    auto reopened = molrs::mrec_writer_open({path.data(), path.size()},
                                            /*flush_every=*/1,
                                            /*durable=*/false);
    EXPECT_EQ(molrs::mrec_writer_committed(*reopened), 4u);
    molrs::mrec_writer_close(std::move(reopened));

    remove_store(path);
}

// ── 3. Frame content ─────────────────────────────────────────────────────

// Frame 1 of a freq-1 run is step 1: the fixture advanced by exactly one
// Velocity-Verlet step (frame 0 is the initial state). Read it back through
// `read_mrec_trajectory_frame` and compare against the same fixture advanced
// one step with no recorder attached — same deterministic recipe, no RNG.
TEST(MrecRecorderTest, StepOneFrameReadsBackCoordinatesSpeciesAndBox) {
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

    auto fr = molrs::read_mrec_trajectory_frame({path.data(), path.size()},
                                                /*index=*/1);
    auto x = molrs::frame_column_f64(*fr, "atoms", "x");
    auto y = molrs::frame_column_f64(*fr, "atoms", "y");
    auto z = molrs::frame_column_f64(*fr, "atoms", "z");
    auto species = molrs::frame_column_u64(*fr, "atoms", "atomic_number");
    auto mass = molrs::frame_column_f64(*fr, "atoms", "mass");
    auto box = molrs::frame_box(*fr).h;

    auto tag = molrs::frame_column_i32(*fr, "atoms", "tag");

    ASSERT_EQ(x.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(y.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(z.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(species.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(mass.size(), static_cast<std::size_t>(kN));
    ASSERT_EQ(tag.size(), static_cast<std::size_t>(kN));
    // The engine sorts rows spatially; the reference frame is matched by tag.
    const double* rx = reference.span(ATV::kAtom_x);
    const double* ry = reference.span(ATV::kAtom_y);
    const double* rz = reference.span(ATV::kAtom_z);
    const TagInt* rtag = reference.span(ATV::kAtom_tag).data();
    for (int i = 0; i < kN; ++i) {
        SCOPED_TRACE("atom " + std::to_string(i));
        const auto k = static_cast<std::size_t>(i);
        int ref = -1;
        for (int j = 0; j < kN; ++j)
            if (rtag[j] == tag[k])
                ref = j;
        ASSERT_GE(ref, 0) << "tag " << tag[k] << " missing from the reference";
        EXPECT_TRUE(std::isfinite(x[k]) && std::isfinite(y[k])
                    && std::isfinite(z[k]));
        EXPECT_NEAR(x[k], rx[ref], 1e-9);
        EXPECT_NEAR(y[k], ry[ref], 1e-9);
        EXPECT_NEAR(z[k], rz[ref], 1e-9);
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

// ── 4. Row identity ──────────────────────────────────────────────────────

// Every frame is written in ascending tag order — a trajectory row names
// the same atom in every frame although the engine re-sorts owned rows at
// each border refresh. The input rows are deliberately listed in reverse
// tag order so a writer that kept the live row order would fail.
TEST(MrecRecorderTest, FramesAreWrittenInTagOrder) {
    const std::string path = fresh_tmp_store("atv_mrec_recorder_order.mrec");
    {
        cpu::Frame frame = make_lj_frame();
        TagInt* tags = frame.span(ATV::kAtom_tag).data();
        for (int i = 0; i < kN; ++i)
            tags[i] = static_cast<TagInt>(kN - i);
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record(cpu::MrecRecorder{path, /*freq=*/2});
        sim.build();
        sim.run(frame, /*n_steps=*/5); // frames at steps 0, 2, 4
    }
    for (std::uint64_t f = 0; f < 3; ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        auto fr =
            molrs::read_mrec_trajectory_frame({path.data(), path.size()}, f);
        auto tag = molrs::frame_column_i32(*fr, "atoms", "tag");
        ASSERT_EQ(tag.size(), static_cast<std::size_t>(kN));
        for (int i = 0; i < kN; ++i)
            EXPECT_EQ(tag[static_cast<std::size_t>(i)], i + 1);
    }
    remove_store(path);
}

// ── 5. Identity columns: exactly what the Frame carries ──────────────────

namespace {

/// @brief Which per-atom identity columns a fixture carries.
enum class Identity { TypeOnly, SpeciesOnly, Both };

/// @brief Eight atoms on a 4 Å line (single-type LJ, which reads no identity
///        column), with only the identity columns `which` names: a
///        coarse-grained frame has the force-field type only, a QM geometry
///        the species only.
cpu::Frame make_identity_frame(Identity which) {
    constexpr int n = 8;
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_x)[i] = 2.0 + 4.0 * static_cast<double>(i);
        frame.span(ATV::kAtom_y)[i] = 5.0;
        frame.span(ATV::kAtom_z)[i] = 5.0;
        frame.span(ATV::kAtom_vx)[i] = 0.001 * static_cast<double>(i);
        frame.span(ATV::kAtom_mass)[i] = 12.0 + static_cast<double>(i % 2);
        frame.span(ATV::kAtom_type)[i] = i % 3;              // 0, 1, 2
        frame.span(ATV::kAtom_atomic_number)[i] = 6 + i % 2; // C, N
    }
    if (which == Identity::SpeciesOnly)
        frame.block("atoms").erase(ATV::kAtom_type.key);
    if (which == Identity::TypeOnly)
        frame.block("atoms").erase(ATV::kAtom_atomic_number.key);
    frame.box = ATV::cpu::Box::cubic(kBox);
    return frame;
}

bool has_atoms_column(const molrs::FrameRef& fr, const char* col) {
    for (const auto& c : molrs::frame_block_columns(fr, "atoms"))
        if (std::string(c.data(), c.size()) == col)
            return true;
    return false;
}

/// @brief Record two free-flight steps of `which` and check frame 0: the
///        store carries exactly the identity columns the Frame had, and
///        `cpu::from_molpy` restores them (absent stays absent).
void check_identity_round_trip(Identity which, const char* store) {
    const std::string path = fresh_tmp_store(store);
    {
        cpu::Frame frame = make_identity_frame(which);
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record(cpu::MrecRecorder{path, /*freq=*/1});
        sim.build();
        sim.run(frame, /*n_steps=*/2);
    }
    const bool want_type = which != Identity::SpeciesOnly;
    const bool want_z = which != Identity::TypeOnly;
    auto fr = molrs::read_mrec_trajectory_frame({path.data(), path.size()}, 0);
    EXPECT_EQ(has_atoms_column(*fr, "type_id"), want_type);
    EXPECT_EQ(has_atoms_column(*fr, "atomic_number"), want_z);
    if (want_type) {
        auto t = molrs::frame_column_u64(*fr, "atoms", "type_id");
        ASSERT_EQ(t.size(), 8u);
        for (std::size_t i = 0; i < 8; ++i)
            EXPECT_EQ(t[i], i % 3) << "row " << i;
    }
    if (want_z) {
        auto z = molrs::frame_column_u64(*fr, "atoms", "atomic_number");
        ASSERT_EQ(z.size(), 8u);
        for (std::size_t i = 0; i < 8; ++i)
            EXPECT_EQ(z[i], 6 + i % 2) << "row " << i;
    }
    // Reading the trajectory back through the strict boundary restores both
    // meanings, each in its own column.
    cpu::Frame back = cpu::from_molpy(*fr);
    ASSERT_EQ(back.nlocal(), 8);
    EXPECT_EQ(back.has("atoms", ATV::kAtom_type.key), want_type);
    EXPECT_EQ(back.has("atoms", ATV::kAtom_atomic_number.key), want_z);
    // Store rows are in tag order, which is the fixture's row order.
    for (int i = 0; i < 8; ++i) {
        if (want_type)
            EXPECT_EQ(back.span(ATV::kAtom_type)[i], i % 3) << "row " << i;
        if (want_z)
            EXPECT_EQ(back.span(ATV::kAtom_atomic_number)[i], 6 + i % 2)
                << "row " << i;
    }
    remove_store(path);
}

/// @brief The store's molrec `status` document (Zarr v3 group attributes).
std::string read_status(const std::string& path) {
    std::ifstream in(std::filesystem::path(path) / "status" / "zarr.json");
    return std::string(std::istreambuf_iterator<char>(in), {});
}

} // namespace

TEST(MrecRecorderTest, TypeOnlyFrameWritesTypeIdAndNoSpecies) {
    check_identity_round_trip(Identity::TypeOnly, "atv_mrec_cpu_type.mrec");
}

TEST(MrecRecorderTest, SpeciesOnlyFrameWritesAtomicNumberAndNoType) {
    check_identity_round_trip(Identity::SpeciesOnly, "atv_mrec_cpu_z.mrec");
}

TEST(MrecRecorderTest, FrameWithBothIdentitiesWritesBoth) {
    check_identity_round_trip(Identity::Both, "atv_mrec_cpu_both.mrec");
}

// ── 6. Run status ────────────────────────────────────────────────────────

// close() ends the store with the molrec status section: the CPU recorder
// appends synchronously, so the run succeeded with every frame written and
// there is no buffer to account for.
TEST(MrecRecorderTest, CloseWritesSucceededStatusWithFrameCounts) {
    const std::string path = fresh_tmp_store("atv_mrec_cpu_status.mrec");
    {
        cpu::Frame frame = make_lj_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record(cpu::MrecRecorder{path, /*freq=*/2});
        sim.build();
        sim.run(frame, /*n_steps=*/5); // frames at steps 0, 2, 4
    }
    const std::string status = read_status(path);
    EXPECT_NE(status.find(R"("node_type":"group")"), std::string::npos)
        << status;
    EXPECT_NE(status.find(R"("state":"succeeded")"), std::string::npos)
        << status;
    EXPECT_NE(status.find(R"("frames_written":3)"), std::string::npos)
        << status;
    EXPECT_EQ(status.find(R"("buffer")"), std::string::npos) << status;
    EXPECT_EQ(status.find(R"("error")"), std::string::npos) << status;
    // The status group sits beside the trajectory without disturbing it.
    auto fr = molrs::read_mrec_trajectory_frame({path.data(), path.size()}, 2);
    EXPECT_EQ(molrs::frame_column_f64(*fr, "atoms", "x").size(),
              static_cast<std::size_t>(kN));
    remove_store(path);
}

// A producer-reported failure and buffer accounting land in the status
// section: state failed, the error message JSON-escaped verbatim, and the
// buffer object; the frames that landed stay readable.
TEST(MrecRecorderTest, WriterRecordsAReportedFailureAndTheBuffer) {
    const std::string path = fresh_tmp_store("atv_mrec_cpu_failed.mrec");
    cpu::Frame frame = make_lj_frame();
    {
        cpu::MrecWriter writer(path, /*flush_every=*/0, /*durable=*/false);
        writer.append(frame, 0, 0.0, false);
        writer.append(frame, 5, 0.0, false);
        writer.set_failed("disk \"full\"\non /x");
        writer.set_failed("a later failure does not replace the first");
        cpu::MrecBufferStatus buffer;
        buffer.budget_bytes = 4096;
        buffer.frame_bytes = 1024;
        buffer.peak_bytes = 3072;
        buffer.blocked_episodes = 7;
        buffer.blocked_seconds = 0.25;
        writer.set_buffer_status(buffer);
        writer.close();
        EXPECT_EQ(writer.committed(), 2u);
    }
    const std::string status = read_status(path);
    EXPECT_NE(status.find(R"("state":"failed")"), std::string::npos) << status;
    EXPECT_NE(status.find(R"("frames_written":2)"), std::string::npos)
        << status;
    EXPECT_NE(status.find(R"("type":"WriteError")"), std::string::npos)
        << status;
    EXPECT_NE(status.find(R"("message":"disk \"full\"\non /x")"),
              std::string::npos)
        << status;
    EXPECT_EQ(status.find("later failure"), std::string::npos) << status;
    EXPECT_NE(status.find(R"("buffer":{"budget_bytes":4096,"frame_bytes":1024,)"
                          R"("peak_bytes":3072,"blocked_episodes":7,)"
                          R"("blocked_seconds":0.25})"),
              std::string::npos)
        << status;
    auto fr = molrs::read_mrec_trajectory_frame({path.data(), path.size()}, 1);
    EXPECT_EQ(molrs::frame_column_f64(*fr, "atoms", "x").size(),
              static_cast<std::size_t>(kN));
    remove_store(path);
}

// An append molpy refuses (a step that does not increase) throws on the
// caller's thread and marks the run failed: the store keeps the frame that
// landed and its status names the store.
TEST(MrecRecorderTest, FailedAppendMarksTheRunFailed) {
    const std::string path = fresh_tmp_store("atv_mrec_cpu_bad_append.mrec");
    cpu::Frame frame = make_lj_frame();
    {
        cpu::MrecWriter writer(path, /*flush_every=*/0, /*durable=*/false);
        writer.append(frame, 3, 0.0, false);
        EXPECT_THROW(writer.append(frame, 3, 0.0, false), std::runtime_error);
        EXPECT_FALSE(writer.failure().empty());
        EXPECT_NE(writer.failure().find(path), std::string::npos)
            << writer.failure();
        writer.close();
        EXPECT_EQ(writer.appended(), 1u);
    }
    const std::string status = read_status(path);
    EXPECT_NE(status.find(R"("state":"failed")"), std::string::npos) << status;
    EXPECT_NE(status.find(R"("frames_written":1)"), std::string::npos)
        << status;
    EXPECT_NE(status.find(R"("type":"WriteError")"), std::string::npos)
        << status;
    remove_store(path);
}

// ── 7. LAMMPS step convention ────────────────────────────────────────────
//
// LAMMPS increments `update->ntimestep` at the top of every step
// (src/verlet.cpp Verlet::run), writes a dump at the end of the step when
// that timestep is a multiple of `every` (src/output.cpp Output::write),
// writes the current state at the start of every run (Output::setup) and
// never writes a timestep twice (`last_dump`). The oracle here is not the
// recorder: each recorded state is compared against an independent driver
// that ran exactly that many steps without one.

namespace {

/// @brief The LJ fixture compressed to 3.5 Å and given a deterministic
///        velocity field, so every step moves every atom by ~1e-3 Å —
///        consecutive states are far apart at the comparison tolerance.
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
    frame.box = ATV::cpu::Box::cubic(kBox);
    return frame;
}

/// @brief Oracle: the moving fixture advanced exactly `steps` steps by a
///        fresh driver with no recorder (`steps = 0`: the setup state).
cpu::Frame state_after(int steps) {
    cpu::Frame frame = make_moving_lj_frame();
    cpu::md::MD sim;
    mount_lj(sim);
    sim.build();
    sim.run(frame, steps);
    return frame;
}

/// @brief Records the step / time axis the recorder hands its writer.
class StepLog : public cpu::MrecRecorder {
  public:
    using cpu::MrecRecorder::MrecRecorder;
    std::vector<std::int64_t> steps;
    std::vector<double> times; ///< [fs]; -1 when the frame had no time

  protected:
    void append_frame(const cpu::Frame& frame, std::int64_t step, double time,
                      bool has_time) override {
        steps.push_back(step);
        times.push_back(has_time ? time : -1.0);
        cpu::MrecRecorder::append_frame(frame, step, time, has_time);
    }
};

/// @brief Store frame `index` holds `ref`'s positions and velocities,
///        atom by atom (matched by tag).
void expect_store_frame_is(const std::string& path, std::uint64_t index,
                           const cpu::Frame& ref) {
    auto fr =
        molrs::read_mrec_trajectory_frame({path.data(), path.size()}, index);
    auto tag = molrs::frame_column_i32(*fr, "atoms", "tag");
    ASSERT_EQ(tag.size(), static_cast<std::size_t>(kN));
    const TagInt* rtag = ref.span(ATV::kAtom_tag).data();
    const std::pair<const char*, const double*> cols[] = {
        {"x", ref.span(ATV::kAtom_x)},   {"y", ref.span(ATV::kAtom_y)},
        {"z", ref.span(ATV::kAtom_z)},   {"vx", ref.span(ATV::kAtom_vx)},
        {"vy", ref.span(ATV::kAtom_vy)}, {"vz", ref.span(ATV::kAtom_vz)}};
    for (const auto& [name, want] : cols) {
        auto got = molrs::frame_column_f64(*fr, "atoms", name);
        for (int i = 0; i < kN; ++i) {
            int r = -1;
            for (int j = 0; j < kN; ++j)
                if (rtag[j] == tag[static_cast<std::size_t>(i)])
                    r = j;
            ASSERT_GE(r, 0);
            EXPECT_NEAR(got[static_cast<std::size_t>(i)], want[r], 1e-10)
                << name << " of tag " << tag[static_cast<std::size_t>(i)];
        }
    }
}

/// @brief Largest |Δx| between two oracle states [Å]: proves the 1e-10
///        comparison above separates consecutive steps.
double max_dx(const cpu::Frame& a, const cpu::Frame& b) {
    double d = 0.0;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            if (a.span(ATV::kAtom_tag)[i] == b.span(ATV::kAtom_tag)[j])
                d = std::max(d, std::abs(a.span(ATV::kAtom_x)[i]
                                         - b.span(ATV::kAtom_x)[j]));
    return d;
}

} // namespace

// A fresh 6-step run at freq 2 records steps 0, 2, 4, 6 — the initial state
// and the final one included — each frame the state after exactly that many
// steps, with time = step · dt.
TEST(MrecRecorderTest,
     FreshRunRecordsZeroThroughNEachTheStateAfterThatManySteps) {
    const std::string path = fresh_tmp_store("atv_mrec_cpu_lammps.mrec");
    StepLog rec{path, /*freq=*/2, /*flush_every=*/0, /*durable=*/false};
    cpu::Frame frame = make_moving_lj_frame();
    {
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        sim.run(frame, /*n_steps=*/6);
    }
    rec.close();
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 2, 4, 6}));
    EXPECT_EQ(rec.times,
              (std::vector<double>{0.0, 2.0 * kDt, 4.0 * kDt, 6.0 * kDt}));
    ASSERT_GT(max_dx(state_after(2), state_after(3)), 1e-5)
        << "the oracle must separate consecutive steps";
    for (std::size_t f = 0; f < rec.steps.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f) + " (step "
                     + std::to_string(rec.steps[f]) + ")");
        expect_store_frame_is(path, f,
                              state_after(static_cast<int>(rec.steps[f])));
    }
    // The last frame is the state run() handed back.
    expect_store_frame_is(path, 3, frame);
    remove_store(path);
}

// Continued `run(cpu::Frame&, …)` calls extend one step axis: each run first
// writes the current state when it is on cadence, but the boundary step
// already written at the end of the previous run is not written again.
// (molpy refuses a non-increasing step, so a duplicate would also throw.)
TEST(MrecRecorderTest, ContinuedRunWritesTheBoundaryStepOnce) {
    const std::string path = fresh_tmp_store("atv_mrec_cpu_continued.mrec");
    StepLog rec{path, /*freq=*/2, /*flush_every=*/0, /*durable=*/false};
    {
        cpu::Frame frame = make_moving_lj_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        sim.run(frame, 4); // 0, 2, 4
        sim.run(frame, 3); // (4 already written), 6 — 7 is off cadence
        sim.run(frame, 1); // 8
    }
    rec.close();
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 2, 4, 6, 8}));
    for (std::size_t f = 0; f < rec.steps.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f) + " (step "
                     + std::to_string(rec.steps[f]) + ")");
        EXPECT_DOUBLE_EQ(rec.times[f], static_cast<double>(rec.steps[f]) * kDt);
        expect_store_frame_is(path, f,
                              state_after(static_cast<int>(rec.steps[f])));
    }
    remove_store(path);
}

// A zero-step run is LAMMPS `run 0`: the setup writes the initial state.
TEST(MrecRecorderTest, ZeroStepRunWritesTheInitialState) {
    const std::string path = fresh_tmp_store("atv_mrec_cpu_run0.mrec");
    StepLog rec{path, /*freq=*/3, /*flush_every=*/0, /*durable=*/false};
    {
        cpu::Frame frame = make_moving_lj_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        sim.run(frame, 0);
    }
    rec.close();
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0}));
    expect_store_frame_is(path, 0, state_after(0));
    remove_store(path);
}
