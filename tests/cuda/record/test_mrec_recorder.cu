/// @file tests/cuda/record/test_mrec_recorder.cu
/// @brief Unit tests for `cuda::MrecRecorder` — the CUDA `*.mrec` sink.
///
/// One concern per test: constructor guards; frame count with step / time
/// axis / cell / images on a free-flight trajectory; CPU↔CUDA column parity
/// of the same two-step LJ run; a held writer never stalls the step loop
/// while the buffer has room; a run on a closed recorder is refused; the
/// LAMMPS step convention (frame `s` is the state after `s` steps, checked
/// against an independent run of exactly `s` steps; steps 0, k, …, n;
/// time = s·dt; a continued run writes its boundary step once); and the
/// lossless backpressure of the driver's shared capture buffer — a
/// deliberately slow writer behind a two-frame and a one-frame budget blocks
/// the run yet every frame lands, each equal to an independent run; a failing
/// writer stops the run promptly with an I/O error and a failed status; a
/// normal run drains completely without blocking; a budget below one frame is
/// refused when the run starts. The
/// recorder is driven through a minimal `cuda::md::MD` because its kernels
/// read the backend's device step and compute stream — there is no smaller
/// way to drive it. Nothing here is a physics claim. Second-scale, tmp
/// stores.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/mrec_recorder.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/record/mrec_recorder.cuh"
#include "atomiverse/molpy.hpp"

using namespace ATV;

namespace {

constexpr double kEps = 0.0104; // [eV]
constexpr double kSigma = 3.4;  // [Å]
constexpr double kCutoff = 6.0; // [Å]
constexpr double kSkin = 0.3;   // [Å]
constexpr double kDt = 1.0;     // [fs]

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

/// @brief Two rows of five interacting argon atoms in a 30 Å box (the CPU
///        recorder test fixture).
cpu::Frame make_lj_frame() {
    constexpr int n = 10;
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.span(kAtom_x)[i] = 8.0 + 3.8 * static_cast<double>(i % 5);
        frame.span(kAtom_y)[i] = 10.0 + 3.8 * static_cast<double>(i / 5);
        frame.span(kAtom_z)[i] = 15.0;
        frame.span(kAtom_vx)[i] = 0.002 * static_cast<double>(i % 3 - 1);
        frame.span(kAtom_vy)[i] = 0.001 * static_cast<double>(i % 2);
        frame.span(kAtom_mass)[i] = 39.948;
        frame.span(kAtom_type)[i] = 18;
        frame.span(kAtom_charge)[i] = 0.1 * static_cast<double>(i % 2) - 0.05;
    }
    frame.box = cpu::Box::cubic(30.0);
    return frame;
}

/// @brief Argon 10 Å apart (no pair within the cutoff) in a 20 Å box;
///        atom 0 starts 1 Å below the +x face moving at 0.5 Å/fs, so it
///        crosses the face within the run and feels no force.
cpu::Frame make_free_flight_frame() {
    constexpr int n = 8;
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.span(kAtom_x)[i] = 4.0 + 10.0 * static_cast<double>(i % 2);
        frame.span(kAtom_y)[i] = 4.0 + 10.0 * static_cast<double>((i / 2) % 2);
        frame.span(kAtom_z)[i] = 4.0 + 10.0 * static_cast<double>(i / 4);
        frame.span(kAtom_mass)[i] = 39.948;
        frame.span(kAtom_type)[i] = 18;
    }
    // Atom 0 flies along x on the line y = z = 9 Å, at least
    // sqrt(5² + 5²) ≈ 7.1 Å from every lattice site (beyond cutoff + skin).
    frame.span(kAtom_x)[0] = 19.0;
    frame.span(kAtom_y)[0] = 9.0;
    frame.span(kAtom_z)[0] = 9.0;
    frame.span(kAtom_vx)[0] = 0.5;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

void mount_cuda_lj(cuda::md::MD& sim) {
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{kCutoff, kSkin}, -10)
        .add(cuda::PairLJCut{kEps, kSigma, kCutoff});
    sim.set_integrator(cuda::md::VelocityVerlet{kDt});
}

void mount_cpu_lj(cpu::md::MD& sim) {
    sim.on(md::Phase::force).add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
}

rust::Box<molrs::FrameRef> read_frame(const std::string& path,
                                      std::uint64_t index) {
    return molrs::read_mrec_trajectory_frame({path.data(), path.size()}, index);
}

/// @brief Owned copy of a bridge vector (rust::Vec has no operator==).
template <typename T>
std::vector<T> as_vector(const rust::Vec<T>& v) {
    return std::vector<T>(v.begin(), v.end());
}

/// @brief The store's molrec `status` document (Zarr v3 group attributes).
std::string read_status(const std::string& path) {
    std::ifstream in(std::filesystem::path(path) / "status" / "zarr.json");
    return std::string(std::istreambuf_iterator<char>(in), {});
}

std::uint64_t committed(const std::string& path) {
    auto w = molrs::mrec_writer_open({path.data(), path.size()},
                                     /*flush_every=*/1, /*durable=*/false);
    const std::uint64_t n = molrs::mrec_writer_committed(*w);
    molrs::mrec_writer_close(std::move(w));
    return n;
}

/// @brief Bytes one captured frame of an MrecRecorder-only driver occupies.
std::uint64_t mrec_frame_bytes(int n) {
    return cuda::CaptureBuffer::frame_bytes(
        n, cuda::MrecRecorder("unused.mrec", 1).capture_columns());
}

/// @brief A recorder whose worker-side sink waits on a gate the test opens,
///        so the writer is held for as long as the test wants.
class HeldRecorder : public cuda::MrecRecorder {
  public:
    HeldRecorder(std::string path, int freq, std::shared_future<void> gate)
        : cuda::MrecRecorder(std::move(path), freq, 0, false),
          gate_(std::move(gate)) {}

    std::atomic<bool> entered{false};
    std::vector<std::int64_t>
        steps;                 ///< written on the worker, read after close
    std::vector<double> times; ///< [fs]; -1 when the frame had no time

  protected:
    void append_frame(const cpu::Frame& frame, std::int64_t step, double time,
                      bool has_time) override {
        entered.store(true);
        // Bounded so a broken recorder fails the test instead of hanging it.
        if (gate_.wait_for(std::chrono::seconds(60))
            != std::future_status::ready)
            throw std::runtime_error("HeldRecorder: gate never opened");
        steps.push_back(step);
        times.push_back(has_time ? time : -1.0);
        cuda::MrecRecorder::append_frame(frame, step, time, has_time);
    }

  private:
    std::shared_future<void> gate_;
};

bool wait_until(const std::atomic<bool>& flag, std::chrono::seconds limit) {
    const auto end = std::chrono::steady_clock::now() + limit;
    while (!flag.load()) {
        if (std::chrono::steady_clock::now() > end)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

} // namespace

// ── 1. Constructor guards ────────────────────────────────────────────────

// Same refusals as cpu::MrecRecorder. The frame buffer is the driver's: a
// zero budget is refused by the driver knob, and the default is the CMake one.
TEST(CudaMrecRecorderTest, ConstructorRejectsBadArguments) {
    EXPECT_THROW(cuda::MrecRecorder("traj.zarr", 1), std::invalid_argument);
    EXPECT_THROW(cuda::MrecRecorder("traj", 1), std::invalid_argument);
    EXPECT_THROW(cuda::MrecRecorder("", 1), std::invalid_argument);
    EXPECT_THROW(cuda::MrecRecorder("traj.mrec", 0), std::invalid_argument);
    EXPECT_NO_THROW(cuda::MrecRecorder("traj.mrec", 1));
    cuda::md::MD sim;
    EXPECT_EQ(sim.capture_buffer_bytes(), ATV::kCaptureBufferBytes);
    EXPECT_THROW(sim.set_capture_buffer_bytes(0), std::invalid_argument);
    // A captured frame holds at least the recorder's 11 double columns,
    // int64 tag and three int32 image flags per atom.
    EXPECT_GE(mrec_frame_bytes(10), 10u * (11 * 8 + 8 + 12));
    EXPECT_THROW(mrec_frame_bytes(0), std::invalid_argument);
}

// ── 2. Count, step axis, cell, images ────────────────────────────────────

// A 9-step run at freq 2 → frames at steps 0, 2, 4, 6, 8 (the initial state
// and the ends of steps 2…8). Atom 0 flies freely through the +x face: every
// frame must give the unwrapped coordinate x + ix·L = x0 + v·step·dt (the
// state after `step` steps), and the frames after
// the crossing carry ix = 1 with x back inside the cell.
TEST(CudaMrecRecorderTest, WritesFramesWithStepCellAndImages) {
    const std::string path = fresh_tmp_store("atv_cuda_mrec_flight.mrec");
    cuda::MrecRecorder rec{path, /*freq=*/2, /*flush_every=*/0,
                           /*durable=*/false};
    {
        cpu::Frame frame = make_free_flight_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        sim.run(frame, /*n_steps=*/9);
        rec.close();
    }
    EXPECT_EQ(rec.n_frames_written(), 5u);
    EXPECT_EQ(rec.n_frames_committed(), 5u);
    ASSERT_EQ(committed(path), 5u);

    bool crossed = false;
    for (std::uint64_t f = 0; f < 5; ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        auto fr = read_frame(path, f);
        // molpy keeps the step on the store's own axis (not echoed as frame
        // meta); the free flight pins which step each frame froze.
        const auto step = static_cast<std::int64_t>(2 * f);
        auto box = molrs::frame_box(*fr).h;
        ASSERT_EQ(box.size(), 9u);
        EXPECT_DOUBLE_EQ(box[0], 20.0);
        EXPECT_DOUBLE_EQ(box[4], 20.0);
        EXPECT_DOUBLE_EQ(box[8], 20.0);
        auto x = molrs::frame_column_f64(*fr, "atoms", "x");
        auto ix = molrs::frame_column_i32(*fr, "atoms", "ix");
        auto tag = molrs::frame_column_i32(*fr, "atoms", "tag");
        ASSERT_EQ(x.size(), 8u);
        ASSERT_EQ(ix.size(), 8u);
        for (std::size_t i = 1; i < tag.size(); ++i)
            EXPECT_LT(tag[i - 1], tag[i]) << "rows are in tag order";
        const double expected = 19.0 + 0.5 * static_cast<double>(step) * kDt;
        EXPECT_NEAR(x[0] + 20.0 * ix[0], expected,
                    sizeof(cuda::pos_t) == 8 ? 1e-9 : 1e-5);
        if (ix[0] != 0) {
            crossed = true;
            EXPECT_EQ(ix[0], 1);
            EXPECT_GE(x[0], 0.0);
            EXPECT_LT(x[0], 20.0);
        }
    }
    EXPECT_TRUE(crossed) << "atom 0 must have been wrapped by the last frame";
    remove_store(path);
}

// ── 3. CPU ↔ CUDA parity ────────────────────────────────────────────────

// The same two-step LJ run recorded on both backends gives the same three
// frames (steps 0, 1, 2):
// tags, species, mass, charges, images and cell exactly; positions,
// velocities, forces and energies within the device precision tier.
TEST(CudaMrecRecorderTest, FramesMatchTheCpuRecorder) {
    const std::string cpu_path = fresh_tmp_store("atv_mrec_parity_cpu.mrec");
    const std::string gpu_path = fresh_tmp_store("atv_mrec_parity_gpu.mrec");
    {
        cpu::Frame frame = make_lj_frame();
        cpu::md::MD sim;
        mount_cpu_lj(sim);
        sim.set_record(cpu::MrecRecorder{cpu_path, 1});
        sim.build();
        sim.run(frame, 2);
    }
    {
        cpu::Frame frame = make_lj_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record(cuda::MrecRecorder{gpu_path, 1});
        sim.build();
        sim.run(frame, 2);
    } // ~MD → destroy() → close()
    ASSERT_EQ(committed(cpu_path), 3u);
    ASSERT_EQ(committed(gpu_path), 3u);

    // FP64 build: integrator rounding only. Mixed: FP32 force math.
    const double tol = sizeof(cuda::force_t) == 8 ? 1e-9 : 1e-4;
    for (std::uint64_t f = 0; f < 3; ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        auto a = read_frame(cpu_path, f);
        auto b = read_frame(gpu_path, f);
        for (const char* col : {"tag", "ix", "iy", "iz"})
            EXPECT_EQ(as_vector(molrs::frame_column_i32(*a, "atoms", col)),
                      as_vector(molrs::frame_column_i32(*b, "atoms", col)))
                << col;
        EXPECT_EQ(
            as_vector(molrs::frame_column_u64(*a, "atoms", "atomic_number")),
            as_vector(molrs::frame_column_u64(*b, "atoms", "atomic_number")));
        EXPECT_EQ(as_vector(molrs::frame_box(*a).h),
                  as_vector(molrs::frame_box(*b).h));
        // mass and charge are device-precision copies of the same inputs.
        for (const char* col : {"x", "y", "z", "vx", "vy", "vz", "fx", "fy",
                                "fz", "mass", "charge"}) {
            SCOPED_TRACE(col);
            auto va = molrs::frame_column_f64(*a, "atoms", col);
            auto vb = molrs::frame_column_f64(*b, "atoms", col);
            ASSERT_EQ(va.size(), vb.size());
            for (std::size_t i = 0; i < va.size(); ++i)
                EXPECT_NEAR(va[i], vb[i], tol) << "row " << i;
        }
        // The per-term breakdown travels too: LJ is all pair energy.
        for (const char* key : {"energy", "energy_pair", "kinetic"})
            EXPECT_NEAR(molrs::frame_get_meta(*a, key).f64_value,
                        molrs::frame_get_meta(*b, key).f64_value, tol)
                << key;
        // The pair virial [eV], evaluated on recorded steps (demand()).
        auto wa = molrs::frame_get_meta(*a, "stress").f64_values;
        auto wb = molrs::frame_get_meta(*b, "stress").f64_values;
        ASSERT_EQ(wa.size(), 6u);
        ASSERT_EQ(wb.size(), 6u);
        for (std::size_t c = 0; c < 6; ++c)
            EXPECT_NEAR(wa[c], wb[c], tol) << "stress[" << c << "]";
    }
    remove_store(cpu_path);
    remove_store(gpu_path);
}

// ── 4. With room in the buffer, a held writer never stalls the loop ─────

// The writer is held inside frame 0 while the run proceeds. With the
// default budget every capture finds room, so no capture ever waits for
// space (zero blocking episodes, the zero-sync fast path); run() returns
// only after the end-of-run drain, once the gate opens, with every frame
// appended in step order.
TEST(CudaMrecRecorderTest, HeldWriterNeverStallsTheLoopWhileTheBufferHasRoom) {
    const std::string path = fresh_tmp_store("atv_cuda_mrec_held.mrec");
    std::promise<void> open;
    HeldRecorder rec{path, /*freq=*/1, open.get_future().share()};
    {
        cpu::Frame frame = make_lj_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        auto running = std::async(std::launch::async,
                                  [&] { sim.run(frame, /*n_steps=*/6); });
        ASSERT_TRUE(wait_until(rec.entered, std::chrono::seconds(30)));
        EXPECT_EQ(rec.n_frames_written(), 0u)
            << "the writer is held inside frame 0";
        // The run cannot report success with frames still waiting.
        EXPECT_EQ(running.wait_for(std::chrono::milliseconds(200)),
                  std::future_status::timeout);
        open.set_value();
        ASSERT_EQ(running.wait_for(std::chrono::seconds(60)),
                  std::future_status::ready);
        running.get();
        EXPECT_EQ(rec.n_frames_written(), 7u) << "run() returned drained";
        const ATV::CaptureStats st = sim.capture_stats();
        EXPECT_EQ(st.blocked_episodes, 0u);
        EXPECT_EQ(st.blocked_seconds, 0.0);
        EXPECT_EQ(st.captures, 7u);
        rec.close();
    }
    EXPECT_EQ(rec.n_frames_committed(), 7u);
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 1, 2, 3, 4, 5, 6}));
    // MD pushed the integrator dt (timestep protocol): time = step · dt.
    EXPECT_EQ(rec.times,
              (std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    ASSERT_EQ(committed(path), 7u);
    remove_store(path);
}

// ── 5. One recorder records one run ──────────────────────────────────────

// Once closed, the recorder has no writer: a further run is refused at its
// first capture (the setup capture), and the store keeps its three frames.
TEST(CudaMrecRecorderTest, RunAfterCloseIsRefused) {
    const std::string path = fresh_tmp_store("atv_cuda_mrec_closed.mrec");
    cuda::MrecRecorder rec{path, /*freq=*/1, 0, false};
    cpu::Frame frame = make_lj_frame();
    cuda::md::MD sim;
    mount_cuda_lj(sim);
    sim.set_record_ref(rec);
    sim.build();
    sim.run(frame, /*n_steps=*/2); // steps 0, 1, 2
    rec.close();
    EXPECT_EQ(rec.n_frames_written(), 3u);
    try {
        sim.run(frame, /*n_steps=*/3);
        FAIL() << "a run on a closed recorder must be refused";
    } catch (const std::logic_error& e) {
        EXPECT_NE(std::string(e.what()).find("closed"), std::string::npos)
            << e.what();
    }
    EXPECT_EQ(committed(path), 3u);
    remove_store(path);
}

// ── 6. Identity columns: exactly what the device frame carries ───────────

namespace {

enum class Identity { TypeOnly, SpeciesOnly, Both };

/// @brief The LJ fixture with only the identity columns `which` names
///        (single-type LJ reads neither): a coarse-grained frame has the
///        force-field type only, a QM geometry the species only.
cpu::Frame make_identity_frame(Identity which) {
    cpu::Frame frame = make_lj_frame();
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_type)[i] = i % 3;
        frame.span(kAtom_atomic_number)[i] = 6 + i % 2;
    }
    if (which == Identity::SpeciesOnly)
        frame.block("atoms").erase(kAtom_type.key);
    if (which == Identity::TypeOnly)
        frame.block("atoms").erase(kAtom_atomic_number.key);
    return frame;
}

bool has_atoms_column(const molrs::FrameRef& fr, const char* col) {
    for (const auto& c : molrs::frame_block_columns(fr, "atoms"))
        if (std::string(c.data(), c.size()) == col)
            return true;
    return false;
}

/// @brief Record two CUDA steps of `which`; every frame carries exactly the
///        identity columns the device frame had — never a type as Z, never
///        a zero-filled species — and `cpu::from_molpy` restores them.
void check_identity_round_trip(Identity which, const char* store) {
    const std::string path = fresh_tmp_store(store);
    {
        cpu::Frame frame = make_identity_frame(which);
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record(cuda::MrecRecorder{path, 1, 0, false});
        sim.build();
        sim.run(frame, 2);
    }
    ASSERT_EQ(committed(path), 3u); // steps 0, 1, 2
    const bool want_type = which != Identity::SpeciesOnly;
    const bool want_z = which != Identity::TypeOnly;
    for (std::uint64_t f = 0; f < 3; ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        auto fr = read_frame(path, f);
        EXPECT_EQ(has_atoms_column(*fr, "type_id"), want_type);
        EXPECT_EQ(has_atoms_column(*fr, "atomic_number"), want_z);
        if (want_type) {
            auto t = molrs::frame_column_u64(*fr, "atoms", "type_id");
            ASSERT_EQ(t.size(), 10u);
            for (std::size_t i = 0; i < t.size(); ++i)
                EXPECT_EQ(t[i], i % 3) << "row " << i;
        }
        if (want_z) {
            auto z = molrs::frame_column_u64(*fr, "atoms", "atomic_number");
            ASSERT_EQ(z.size(), 10u);
            for (std::size_t i = 0; i < z.size(); ++i)
                EXPECT_EQ(z[i], 6 + i % 2) << "row " << i;
        }
        cpu::Frame back = cpu::from_molpy(*fr);
        ASSERT_EQ(back.nlocal(), 10);
        EXPECT_EQ(back.has("atoms", kAtom_type.key), want_type);
        EXPECT_EQ(back.has("atoms", kAtom_atomic_number.key), want_z);
        for (int i = 0; i < 10; ++i) {
            if (want_type)
                EXPECT_EQ(back.span(kAtom_type)[i], i % 3) << "row " << i;
            if (want_z)
                EXPECT_EQ(back.span(kAtom_atomic_number)[i], 6 + i % 2)
                    << "row " << i;
        }
    }
    remove_store(path);
}

} // namespace

TEST(CudaMrecRecorderTest, TypeOnlyFrameWritesTypeIdAndNoSpecies) {
    check_identity_round_trip(Identity::TypeOnly, "atv_cuda_mrec_type.mrec");
}

TEST(CudaMrecRecorderTest, SpeciesOnlyFrameWritesAtomicNumberAndNoType) {
    check_identity_round_trip(Identity::SpeciesOnly, "atv_cuda_mrec_z.mrec");
}

TEST(CudaMrecRecorderTest, FrameWithBothIdentitiesWritesBoth) {
    check_identity_round_trip(Identity::Both, "atv_cuda_mrec_both.mrec");
}

// ── 7. LAMMPS step convention ────────────────────────────────────────────
//
// LAMMPS increments `update->ntimestep` at the top of every step
// (src/verlet.cpp Verlet::run), dumps at the end of a step whose timestep is
// a multiple of `every` (src/output.cpp Output::write), dumps the current
// state at the start of every run (Output::setup) and never writes a
// timestep twice (`last_dump`). The oracle is not the recorder: each frame
// is compared with an independent CUDA run of exactly that many steps.

namespace {

/// @brief Ten argon atoms 3.5 Å apart with a deterministic velocity field:
///        every step moves every atom by ~1e-3 Å.
cpu::Frame make_moving_lj_frame() {
    constexpr int n = 10;
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        frame.span(kAtom_x)[i] = 8.0 + 3.5 * static_cast<double>(i % 5);
        frame.span(kAtom_y)[i] = 10.0 + 3.5 * static_cast<double>(i / 5);
        frame.span(kAtom_z)[i] = 15.0;
        frame.span(kAtom_vx)[i] = 0.002 * static_cast<double>(i % 3 - 1);
        frame.span(kAtom_vy)[i] = 0.001 * static_cast<double>(i % 2);
        frame.span(kAtom_vz)[i] = 0.0005 * static_cast<double>(i % 4 - 2);
        frame.span(kAtom_mass)[i] = 39.948;
        frame.span(kAtom_type)[i] = 18;
        frame.span(kAtom_atomic_number)[i] = 18;
        frame.span(kAtom_tag)[i] = static_cast<TagInt>(i + 1);
    }
    frame.box = cpu::Box::cubic(30.0);
    return frame;
}

/// @brief Oracle: the moving fixture after exactly `steps` CUDA steps of a
///        fresh driver with no recorder (`steps = 0`: the setup state).
cpu::Frame cuda_state_after(int steps) {
    cpu::Frame frame = make_moving_lj_frame();
    cuda::md::MD sim;
    mount_cuda_lj(sim);
    sim.build();
    sim.run(frame, steps);
    return frame;
}

/// @brief Logs the step / time axis on the worker; no gate.
class StepLog : public cuda::MrecRecorder {
  public:
    StepLog(std::string path, int freq)
        : cuda::MrecRecorder(std::move(path), freq, 0, false) {}
    std::vector<std::int64_t>
        steps;                 ///< written on the worker, read after close
    std::vector<double> times; ///< [fs]; -1 when the frame had no time

  protected:
    void append_frame(const cpu::Frame& frame, std::int64_t step, double time,
                      bool has_time) override {
        steps.push_back(step);
        times.push_back(has_time ? time : -1.0);
        cuda::MrecRecorder::append_frame(frame, step, time, has_time);
    }
};

/// @brief Store frame `index` holds `ref`'s positions and velocities, atom
///        by atom (matched by tag; both runs are the same device code).
void expect_store_frame_is(const std::string& path, std::uint64_t index,
                           const cpu::Frame& ref) {
    const double tol = sizeof(cuda::pos_t) == 8 ? 1e-10 : 1e-5;
    auto fr = read_frame(path, index);
    auto tag = molrs::frame_column_i32(*fr, "atoms", "tag");
    ASSERT_EQ(tag.size(), static_cast<std::size_t>(ref.nlocal()));
    const auto* rtag = ref.span(kAtom_tag).data();
    const std::pair<const char*, const double*> cols[] = {
        {"x", ref.span(kAtom_x)},   {"y", ref.span(kAtom_y)},
        {"z", ref.span(kAtom_z)},   {"vx", ref.span(kAtom_vx)},
        {"vy", ref.span(kAtom_vy)}, {"vz", ref.span(kAtom_vz)}};
    for (const auto& [name, want] : cols) {
        auto got = molrs::frame_column_f64(*fr, "atoms", name);
        for (std::size_t i = 0; i < tag.size(); ++i) {
            int r = -1;
            for (int j = 0; j < ref.nlocal(); ++j)
                if (rtag[j] == tag[i])
                    r = j;
            ASSERT_GE(r, 0);
            EXPECT_NEAR(got[i], want[r], tol) << name << " of tag " << tag[i];
        }
    }
}

} // namespace

// A fresh 6-step run at freq 2 records steps 0, 2, 4, 6 — the initial and
// the final state included — each the state after exactly that many steps,
// with time = step · dt.
TEST(CudaMrecRecorderTest,
     FreshRunRecordsZeroThroughNEachTheStateAfterThatManySteps) {
    const std::string path = fresh_tmp_store("atv_cuda_mrec_lammps.mrec");
    StepLog rec{path, /*freq=*/2};
    cpu::Frame frame = make_moving_lj_frame();
    {
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        sim.run(frame, /*n_steps=*/6);
        rec.close();
    }
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 2, 4, 6}));
    EXPECT_EQ(rec.times,
              (std::vector<double>{0.0, 2.0 * kDt, 4.0 * kDt, 6.0 * kDt}));
    {
        // The oracle separates consecutive steps far beyond the tolerance.
        const cpu::Frame a = cuda_state_after(2), b = cuda_state_after(3);
        double d = 0.0;
        for (int i = 0; i < a.nlocal(); ++i)
            for (int j = 0; j < b.nlocal(); ++j)
                if (a.span(kAtom_tag)[i] == b.span(kAtom_tag)[j])
                    d = std::max(
                        d, std::abs(a.span(kAtom_x)[i] - b.span(kAtom_x)[j]));
        ASSERT_GT(d, 1e-4);
    }
    for (std::size_t f = 0; f < rec.steps.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f) + " (step "
                     + std::to_string(rec.steps[f]) + ")");
        expect_store_frame_is(path, f,
                              cuda_state_after(static_cast<int>(rec.steps[f])));
    }
    expect_store_frame_is(path, 3, frame); // the state run() handed back
    remove_store(path);
}

// Continued run() calls on the same frame extend one step axis; the
// boundary step written at the end of a run is not written again by the
// next run's setup (LAMMPS `last_dump`), whether or not the step is on
// cadence. molpy refuses a non-increasing step, so a duplicate would also
// fail the worker.
TEST(CudaMrecRecorderTest, ContinuedRunWritesTheBoundaryStepOnce) {
    const std::string path = fresh_tmp_store("atv_cuda_mrec_continued.mrec");
    StepLog rec{path, /*freq=*/2};
    {
        cpu::Frame frame = make_moving_lj_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        sim.run(frame, 4); // 0, 2, 4
        sim.run(frame, 3); // (4 already written), 6 — 7 is off cadence
        sim.run(frame, 1); // 8
        rec.close();
    }
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 2, 4, 6, 8}));
    for (std::size_t f = 0; f < rec.steps.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f) + " (step "
                     + std::to_string(rec.steps[f]) + ")");
        EXPECT_DOUBLE_EQ(rec.times[f], static_cast<double>(rec.steps[f]) * kDt);
        expect_store_frame_is(path, f,
                              cuda_state_after(static_cast<int>(rec.steps[f])));
    }
    remove_store(path);
}

// ── 8. Lossless backpressure under a byte budget ─────────────────────────
//
// The oracles are independent of the recorder: the step axis must be exactly
// 0, k, …, n, each frame must equal an independent CUDA run of exactly that
// many steps (`cuda_state_after`), and blocking is read from the store's own
// status section as well as from the recorder.

namespace {

/// @brief A deliberately slow writer: every append sleeps `delay` first.
class SlowRecorder : public cuda::MrecRecorder {
  public:
    SlowRecorder(std::string path, int freq, std::chrono::milliseconds delay)
        : cuda::MrecRecorder(std::move(path), freq, 0, false), delay_(delay) {}
    std::vector<std::int64_t>
        steps; ///< written on the worker, read after close

  protected:
    void append_frame(const cpu::Frame& frame, std::int64_t step, double time,
                      bool has_time) override {
        std::this_thread::sleep_for(delay_);
        steps.push_back(step);
        cuda::MrecRecorder::append_frame(frame, step, time, has_time);
    }

  private:
    std::chrono::milliseconds delay_;
};

/// @brief A writer that fails on its `fail_at`-th append (0-based), as a full
///        disk would, after a short per-frame delay.
class FailingRecorder : public cuda::MrecRecorder {
  public:
    FailingRecorder(std::string path, int fail_at)
        : cuda::MrecRecorder(std::move(path), /*freq=*/1, 0, false),
          fail_at_(fail_at) {}

  protected:
    void append_frame(const cpu::Frame& frame, std::int64_t step, double time,
                      bool has_time) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (appends_++ == fail_at_)
            throw std::runtime_error("injected: No space left on device");
        cuda::MrecRecorder::append_frame(frame, step, time, has_time);
    }

  private:
    int fail_at_;
    int appends_ = 0;
};

/// @brief The integer or real value of `"key":` in a status document.
double status_number(const std::string& status, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const auto at = status.find(needle);
    if (at == std::string::npos)
        throw std::runtime_error("status has no " + key + ": " + status);
    return std::strtod(status.c_str() + at + needle.size(), nullptr);
}

/// @brief Record `n_steps` at `every` behind a `frames`-frame budget with a
///        writer that takes 20 ms per frame (the steps take microseconds),
///        then check that nothing was lost and that the run did block.
void check_slow_writer_loses_nothing(std::uint64_t frames, const char* store) {
    constexpr int kEvery = 2;
    constexpr int kSteps = 10;
    const std::string path = fresh_tmp_store(store);
    const std::uint64_t frame = mrec_frame_bytes(10);
    const std::uint64_t budget = frames * frame;
    SlowRecorder rec{path, kEvery, std::chrono::milliseconds(20)};
    ATV::CaptureStats st;
    {
        cpu::Frame state = make_moving_lj_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record_ref(rec);
        sim.set_capture_buffer_bytes(budget);
        sim.build();
        sim.run(state, kSteps);
        // Normal completion drained the buffer before run() returned.
        EXPECT_EQ(rec.n_frames_written(), 6u);
        st = sim.capture_stats();
        rec.close();
    }
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 2, 4, 6, 8, 10}));
    ASSERT_EQ(committed(path), 6u);
    for (std::size_t f = 0; f < rec.steps.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f) + " (step "
                     + std::to_string(rec.steps[f]) + ")");
        expect_store_frame_is(path, f,
                              cuda_state_after(static_cast<int>(rec.steps[f])));
    }
    EXPECT_GT(st.blocked_episodes, 0u);
    EXPECT_GT(st.blocked_seconds, 0.0);
    EXPECT_EQ(st.peak_bytes, budget) << "the buffer filled";
    const std::string status = read_status(path);
    EXPECT_NE(status.find(R"("state":"succeeded")"), std::string::npos)
        << status;
    EXPECT_NE(status.find(R"("frames_written":6)"), std::string::npos)
        << status;
    EXPECT_EQ(status_number(status, "budget_bytes"),
              static_cast<double>(budget));
    EXPECT_EQ(status_number(status, "frame_bytes"), static_cast<double>(frame));
    EXPECT_EQ(status_number(status, "peak_bytes"), static_cast<double>(budget));
    EXPECT_EQ(status_number(status, "blocked_episodes"),
              static_cast<double>(st.blocked_episodes));
    EXPECT_GT(status_number(status, "blocked_seconds"), 0.0);
    remove_store(path);
}

} // namespace

TEST(CudaMrecRecorderTest,
     SlowWriterBehindATwoFrameBudgetBlocksButLosesNothing) {
    check_slow_writer_loses_nothing(2, "atv_cuda_mrec_slow2.mrec");
}

TEST(CudaMrecRecorderTest,
     SlowWriterBehindAOneFrameBudgetBlocksButLosesNothing) {
    check_slow_writer_loses_nothing(1, "atv_cuda_mrec_slow1.mrec");
}

// The writer fails on its third frame while a 100000-step run is under way
// behind a two-frame budget. The run must stop promptly — it would need
// > 500 s to finish at 5 ms per frame — with an I/O error naming the store
// and the cause, not wait for space that will never free; the store keeps
// the two frames written before the failure and its status says failed.
TEST(CudaMrecRecorderTest, WriterFailureStopsTheRunPromptly) {
    const std::string path = fresh_tmp_store("atv_cuda_mrec_fail.mrec");
    FailingRecorder rec{path, /*fail_at=*/2};
    {
        cpu::Frame frame = make_lj_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record_ref(rec);
        sim.set_capture_buffer_bytes(2 * mrec_frame_bytes(10));
        sim.build();
        auto running = std::async(std::launch::async, [&]() -> std::string {
            try {
                sim.run(frame, /*n_steps=*/100000);
            } catch (const std::runtime_error& e) {
                return e.what();
            }
            return {};
        });
        if (running.wait_for(std::chrono::seconds(60))
            != std::future_status::ready) {
            std::fprintf(stderr, "the run hung after the writer failed\n");
            std::_Exit(1);
        }
        const std::string what = running.get();
        ASSERT_FALSE(what.empty()) << "a failed writer must stop the run";
        EXPECT_NE(what.find(path), std::string::npos) << what;
        EXPECT_NE(what.find("No space left on device"), std::string::npos)
            << what;
        EXPECT_NE(what.find("failed at step 2"), std::string::npos) << what;
        // Already raised by the run: close() commits without rethrowing.
        EXPECT_NO_THROW(rec.close());
    }
    EXPECT_EQ(rec.n_frames_written(), 2u);
    ASSERT_EQ(committed(path), 2u);
    const std::string status = read_status(path);
    EXPECT_NE(status.find(R"("state":"failed")"), std::string::npos) << status;
    EXPECT_NE(status.find(R"("type":"WriteError")"), std::string::npos)
        << status;
    EXPECT_NE(status.find("No space left on device"), std::string::npos)
        << status;
    EXPECT_NE(status.find(R"("frames_written":2)"), std::string::npos)
        << status;
    remove_store(path);
}

// A run with room to spare drains completely before run() returns, never
// blocks, and its status records the buffer it used.
TEST(CudaMrecRecorderTest, NormalRunDrainsWithoutBlocking) {
    const std::string path = fresh_tmp_store("atv_cuda_mrec_ok.mrec");
    cuda::MrecRecorder rec{path, /*freq=*/2, 0, false};
    {
        cpu::Frame frame = make_lj_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.set_record_ref(rec);
        sim.build();
        EXPECT_NO_THROW(sim.run(frame, /*n_steps=*/6)); // steps 0, 2, 4, 6
        EXPECT_EQ(rec.n_frames_written(), 4u) << "drained at the run's end";
        const ATV::CaptureStats st = sim.capture_stats();
        EXPECT_EQ(st.blocked_episodes, 0u);
        EXPECT_EQ(st.blocked_seconds, 0.0);
        EXPECT_EQ(st.captures, 4u);
        EXPECT_EQ(st.frame_bytes, mrec_frame_bytes(10));
        EXPECT_LE(st.peak_bytes, 4 * mrec_frame_bytes(10));
        rec.close();
    }
    const std::string status = read_status(path);
    EXPECT_NE(status.find(R"("state":"succeeded")"), std::string::npos)
        << status;
    EXPECT_NE(status.find(R"("frames_written":4)"), std::string::npos)
        << status;
    EXPECT_EQ(status_number(status, "budget_bytes"),
              static_cast<double>(ATV::kCaptureBufferBytes));
    EXPECT_EQ(status_number(status, "blocked_episodes"), 0.0);
    EXPECT_EQ(status_number(status, "blocked_seconds"), 0.0);
    EXPECT_EQ(status.find(R"("error")"), std::string::npos) << status;
    remove_store(path);
}

// A budget one byte short of a frame cannot hold any frame: the driver
// refuses it when the run starts, naming both sizes.
TEST(CudaMrecRecorderTest, BudgetBelowOneFrameIsRefusedAtBuild) {
    const std::string path = fresh_tmp_store("atv_cuda_mrec_tiny.mrec");
    const std::uint64_t frame = mrec_frame_bytes(10);
    cuda::MrecRecorder rec{path, /*freq=*/1, 0, false};
    cpu::Frame state = make_lj_frame();
    cuda::md::MD sim;
    mount_cuda_lj(sim);
    sim.set_record_ref(rec);
    sim.set_capture_buffer_bytes(frame - 1);
    sim.build();
    try {
        sim.run(state, 2);
        FAIL() << "a budget below one frame must be refused";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(
            what.find("capture_buffer_bytes = " + std::to_string(frame - 1)),
            std::string::npos)
            << what;
        EXPECT_NE(what.find("needs " + std::to_string(frame) + " bytes"),
                  std::string::npos)
            << what;
    }
    EXPECT_FALSE(std::filesystem::exists(path));
}
