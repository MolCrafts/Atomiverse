/// @file tests/cuda/asynchronous/test_capture_buffer.cu
/// @brief The CUDA driver's shared, lossless capture buffer
///        (`cuda::CaptureBuffer` + `cuda::AsyncConsumer`).
///
/// Oracles independent of the code under test: every delivered frame is
/// compared with a fresh CUDA run of exactly that many steps and no consumer
/// (`state_after`); expected step lists are written out by hand from the
/// LAMMPS output convention (initial state, every k-th completed step, never
/// twice); the XYZ file is read back from disk. The MD step is graph-replayed
/// (the consumers' captures are graph-captured), so these runs exercise the
/// zero-sync fast path and the host-side admission together.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/asynchronous/async_consumer.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/record/recorder.cuh"
#ifdef ATV_HAS_IO
#include "atomiverse/cuda/record/mrec_recorder.cuh"
#include "atomiverse/molpy.hpp"
#endif

using namespace ATV;
using namespace std::chrono_literals;

namespace {

constexpr int kN = 10;
constexpr double kEps = 0.0104; // [eV]
constexpr double kSigma = 3.4;  // [Å]
constexpr double kCutoff = 6.0; // [Å]
constexpr double kSkin = 0.3;   // [Å]
constexpr double kDt = 1.0;     // [fs]
constexpr double kBox = 30.0;   // [Å]

/// @brief Ten argon atoms 3.5 Å apart (repulsive: every step moves them)
///        with a deterministic velocity field.
cpu::Frame make_frame() {
    cpu::Frame frame(kN);
    for (int i = 0; i < kN; ++i) {
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
    frame.box = cpu::Box::cubic(kBox);
    return frame;
}

void mount_lj(cuda::md::MD& sim) {
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{kCutoff, kSkin}, -10)
        .add(cuda::PairLJCut{kEps, kSigma, kCutoff});
    sim.set_integrator(cuda::md::VelocityVerlet{kDt});
}

/// x by tag of one state [Å].
using Positions = std::map<std::int64_t, double>;

/// @brief Oracle: the fixture after exactly `steps` steps of a fresh driver
///        with no consumer.
Positions state_after(int steps) {
    cpu::Frame frame = make_frame();
    cuda::md::MD sim;
    mount_lj(sim);
    sim.build();
    sim.run(frame, steps);
    Positions p;
    for (int i = 0; i < frame.nlocal(); ++i)
        p[frame.span(kAtom_tag)[i]] = frame.span(kAtom_x)[i];
    return p;
}

double tolerance() { return sizeof(cuda::pos_t) == 8 ? 1e-9 : 1e-5; }

void expect_positions(const Positions& got, const Positions& want) {
    ASSERT_EQ(got.size(), want.size());
    for (const auto& [tag, x] : want)
        EXPECT_NEAR(got.at(tag), x, tolerance()) << "tag " << tag;
}

/// @brief Test consumer (an analysis observer): keeps (step, x by tag);
///        optionally slow, optionally failing at one step.
class Probe : public cuda::AsyncObserver {
  public:
    Probe(std::string name, int every, std::chrono::milliseconds delay = 0ms,
          BigInt fail_at = -1)
        : name_(std::move(name)), every_(every), delay_(delay),
          fail_at_(fail_at) {}

    std::string name() const override { return name_; }
    int frequency() const override { return every_; }
    CaptureColumns capture_columns() const override {
        CaptureColumns c;
        c.positions = c.velocities = c.tags = true;
        return c;
    }

    std::mutex mtx;
    std::vector<BigInt> steps;
    std::vector<Positions> frames;

  protected:
    void consume(const cuda::CaptureView& v) override {
        if (delay_.count() > 0)
            std::this_thread::sleep_for(delay_);
        if (v.step == fail_at_)
            throw std::runtime_error("injected: cannot write 'probe.out'");
        Positions p;
        for (int i = 0; i < v.n_atoms; ++i)
            p[v.tag[i]] = v.x[i];
        std::lock_guard<std::mutex> lk(mtx);
        steps.push_back(v.step);
        frames.push_back(std::move(p));
    }

  private:
    std::string name_;
    int every_;
    std::chrono::milliseconds delay_;
    BigInt fail_at_;
};

void expect_oracle_frames(Probe& probe) {
    ASSERT_EQ(probe.steps.size(), probe.frames.size());
    for (std::size_t f = 0; f < probe.steps.size(); ++f) {
        SCOPED_TRACE(probe.name() + " step " + std::to_string(probe.steps[f]));
        expect_positions(probe.frames[f],
                         state_after(static_cast<int>(probe.steps[f])));
    }
}

std::vector<BigInt> every_k(int k, int n) {
    std::vector<BigInt> v;
    for (int s = 0; s <= n; s += k)
        v.push_back(s);
    return v;
}

CaptureColumns probe_columns() {
    CaptureColumns c;
    c.positions = c.velocities = c.tags = true;
    return c;
}

std::string fresh_tmp(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    return p.string();
}

/// @brief x of every XYZ frame, rows in file (device) order.
std::vector<std::vector<double>> read_xyz_x(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> frames;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        const int n = std::stoi(line);
        std::getline(in, line); // comment
        std::vector<double> xs;
        for (int i = 0; i < n; ++i) {
            std::getline(in, line);
            std::istringstream ss(line);
            std::string symbol;
            double x = 0, y = 0, z = 0;
            ss >> symbol >> x >> y >> z;
            xs.push_back(x);
        }
        frames.push_back(std::move(xs));
    }
    return frames;
}

} // namespace

// The oracle separates consecutive steps far beyond the tolerances above.
TEST(CudaCaptureBuffer, OracleSeparatesConsecutiveSteps) {
    const Positions a = state_after(2), b = state_after(3);
    double dx = 0.0;
    for (const auto& [tag, x] : a)
        dx = std::max(dx, std::abs(x - b.at(tag)));
    EXPECT_GT(dx, 1e-4);
}

// Two recorders (`MrecRecorder`, `XyzRecorder`) and one analysis observer on
// the same cadence share ONE capture per step: a one-frame budget suffices,
// the buffer accounts one frame per captured step (not three), and each
// consumer sees the state of an independent run of that many steps.
TEST(CudaCaptureBuffer, TwoRecordersAndAnObserverCaptureOncePerStep) {
    constexpr int n = 8, k = 2;
    const std::string xyz = fresh_tmp("atv_cuda_capture_shared.xyz");
    Probe probe{"probe", k};
    cuda::XyzRecorder xyz_rec{xyz, k};
    CaptureColumns columns = probe_columns() | xyz_rec.capture_columns();
#ifdef ATV_HAS_IO
    const std::string store = fresh_tmp("atv_cuda_capture_shared.mrec");
    cuda::MrecRecorder mrec{store, k, 0, false};
    columns = columns | mrec.capture_columns();
#endif
    const std::uint64_t frame = cuda::CaptureBuffer::frame_bytes(kN, columns);
    CaptureStats st;
    {
        cpu::Frame f = make_frame();
        cuda::md::MD sim;
        mount_lj(sim);
        sim.on(md::Phase::end_of_step).add_ref(xyz_rec, +10);
        sim.on(md::Phase::end_of_step).add_ref(probe, +20);
#ifdef ATV_HAS_IO
        sim.set_record_ref(mrec);
#endif
        sim.set_capture_buffer_bytes(frame);
        sim.build();
        sim.run(f, n);
        st = sim.capture_stats();
#ifdef ATV_HAS_IO
        EXPECT_EQ(mrec.n_frames_written(), 5u);
        mrec.close();
#endif
    }
    const std::vector<BigInt> want{0, 2, 4, 6, 8};
    EXPECT_EQ(probe.steps, want);
    EXPECT_EQ(st.frame_bytes, frame);
    EXPECT_EQ(st.capacity, 1u);
    EXPECT_EQ(st.captures, want.size()) << "one capture per step, not three";
    EXPECT_EQ(st.peak_bytes, frame) << "never more than one frame in flight";
    expect_oracle_frames(probe);
    // The XYZ rows are in device order (the device may sort atoms), so the
    // frame's x values are compared with the oracle's as sorted multisets.
    const auto frames = read_xyz_x(xyz);
    ASSERT_EQ(frames.size(), want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        const Positions oracle = state_after(static_cast<int>(want[i]));
        std::vector<double> got = frames[i], ref;
        for (const auto& [tag, x] : oracle)
            ref.push_back(x);
        std::sort(got.begin(), got.end());
        std::sort(ref.begin(), ref.end());
        ASSERT_EQ(got.size(), ref.size());
        for (std::size_t r = 0; r < ref.size(); ++r)
            EXPECT_NEAR(got[r], ref[r], 1e-5) << "xyz frame " << i;
    }
#ifdef ATV_HAS_IO
    for (std::uint64_t i = 0; i < want.size(); ++i) {
        auto fr =
            molrs::read_mrec_trajectory_frame({store.data(), store.size()}, i);
        auto xs = molrs::frame_column_f64(*fr, "atoms", "x");
        auto tags = molrs::frame_column_i32(*fr, "atoms", "tag");
        const Positions oracle = state_after(static_cast<int>(want[i]));
        ASSERT_EQ(xs.size(), static_cast<std::size_t>(kN));
        for (std::size_t r = 0; r < xs.size(); ++r)
            EXPECT_NEAR(xs[r], oracle.at(static_cast<std::int64_t>(tags[r])),
                        tolerance())
                << "mrec frame " << i;
    }
    std::error_code ec;
    std::filesystem::remove_all(store, ec);
#endif
    std::filesystem::remove(xyz);
}

namespace {

/// A slow and a fast consumer on one buffer of `frames` frames, every step.
void slow_consumer_loses_nothing(std::uint64_t frames) {
    constexpr int n = 12;
    Probe slow{"slow", 1, 5ms};
    Probe fast{"fast", 1};
    const std::uint64_t frame =
        cuda::CaptureBuffer::frame_bytes(kN, probe_columns());
    CaptureStats st;
    {
        cpu::Frame f = make_frame();
        cuda::md::MD sim;
        mount_lj(sim);
        sim.on(md::Phase::end_of_step).add_ref(slow, +20);
        sim.on(md::Phase::end_of_step).add_ref(fast, +20);
        sim.set_capture_buffer_bytes(frames * frame);
        sim.build();
        sim.run(f, n);
        st = sim.capture_stats();
        // Drained before run() returned.
        EXPECT_EQ(slow.steps.size(), static_cast<std::size_t>(n + 1));
    }
    EXPECT_EQ(slow.steps, every_k(1, n));
    EXPECT_EQ(fast.steps, every_k(1, n));
    EXPECT_EQ(st.capacity, frames);
    EXPECT_EQ(st.captures, static_cast<std::uint64_t>(n + 1));
    EXPECT_EQ(st.peak_bytes, frames * frame) << "the buffer filled";
    EXPECT_GT(st.blocked_episodes, 0u);
    EXPECT_GT(st.blocked_seconds, 0.0);
    expect_oracle_frames(slow);
    expect_oracle_frames(fast);
}

} // namespace

TEST(CudaCaptureBuffer,
     SlowConsumerBehindAOneFrameBudgetBlocksButLosesNothing) {
    slow_consumer_loses_nothing(1);
}

TEST(CudaCaptureBuffer,
     SlowConsumerBehindATwoFrameBudgetBlocksButLosesNothing) {
    slow_consumer_loses_nothing(2);
}

// Consumers on different cadences share the captures where they coincide
// and each receives exactly its own steps.
TEST(CudaCaptureBuffer, DifferentCadencesShareCoincidentCaptures) {
    Probe two{"two", 2}, three{"three", 3};
    CaptureStats st;
    {
        cpu::Frame f = make_frame();
        cuda::md::MD sim;
        mount_lj(sim);
        sim.on(md::Phase::end_of_step).add_ref(two, +20);
        sim.on(md::Phase::end_of_step).add_ref(three, +20);
        sim.build();
        sim.run(f, 12);
        st = sim.capture_stats();
    }
    EXPECT_EQ(two.steps, every_k(2, 12));
    EXPECT_EQ(three.steps, every_k(3, 12));
    // Union of {0,2,…,12} and {0,3,…,12}: 0 2 3 4 6 8 9 10 12.
    EXPECT_EQ(st.captures, 9u);
    EXPECT_EQ(st.blocked_episodes, 0u);
    expect_oracle_frames(two);
    expect_oracle_frames(three);
}

// Continued runs extend one series: the boundary step is delivered once,
// and a normal run drains without blocking.
TEST(CudaCaptureBuffer, ContinuedRunDeliversTheBoundaryStepOnce) {
    Probe probe{"probe", 2};
    CaptureStats st;
    {
        cpu::Frame f = make_frame();
        cuda::md::MD sim;
        mount_lj(sim);
        sim.on(md::Phase::end_of_step).add_ref(probe, +20);
        sim.build();
        sim.run(f, 4);
        EXPECT_EQ(probe.steps, every_k(2, 4))
            << "drained before run() returned";
        sim.run(f, 4);
        st = sim.capture_stats();
    }
    EXPECT_EQ(probe.steps, every_k(2, 8));
    EXPECT_EQ(st.captures, 5u);
    EXPECT_EQ(st.blocked_episodes, 0u);
    EXPECT_EQ(st.blocked_seconds, 0.0);
    expect_oracle_frames(probe);
}

// A budget below one captured frame is refused when the run starts, naming
// both sizes.
TEST(CudaCaptureBuffer, BudgetBelowOneFrameIsRefused) {
    const std::uint64_t frame =
        cuda::CaptureBuffer::frame_bytes(kN, probe_columns());
    Probe probe{"probe", 1};
    cpu::Frame f = make_frame();
    cuda::md::MD sim;
    mount_lj(sim);
    sim.on(md::Phase::end_of_step).add_ref(probe, +20);
    sim.set_capture_buffer_bytes(frame - 1);
    sim.build();
    try {
        sim.run(f, 2);
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
    EXPECT_TRUE(probe.steps.empty());
}

// A consumer that fails mid-run stops the run promptly with an error naming
// the consumer, the step and the cause; the producer never waits on space
// that will not free (one-frame budget). Bounded by a watchdog.
TEST(CudaCaptureBuffer, ConsumerFailureStopsTheRunPromptly) {
    Probe failing{"failing_probe", 1, 1ms, /*fail_at=*/3};
    Probe fast{"fast", 1};
    const std::uint64_t frame =
        cuda::CaptureBuffer::frame_bytes(kN, probe_columns());
    {
        cpu::Frame f = make_frame();
        cuda::md::MD sim;
        mount_lj(sim);
        sim.on(md::Phase::end_of_step).add_ref(failing, +20);
        sim.on(md::Phase::end_of_step).add_ref(fast, +20);
        sim.set_capture_buffer_bytes(frame);
        sim.build();
        auto running = std::async(std::launch::async, [&]() -> std::string {
            try {
                sim.run(f, 100000);
            } catch (const std::runtime_error& e) {
                return e.what();
            }
            return {};
        });
        if (running.wait_for(60s) != std::future_status::ready) {
            std::fprintf(stderr, "the run hung after a consumer failed\n");
            std::_Exit(1);
        }
        const std::string what = running.get();
        ASSERT_FALSE(what.empty()) << "a failed consumer must stop the run";
        EXPECT_NE(what.find("cuda::failing_probe"), std::string::npos) << what;
        EXPECT_NE(what.find("cannot write 'probe.out'"), std::string::npos)
            << what;
        EXPECT_NE(what.find("failed at step 3"), std::string::npos) << what;
    }
    EXPECT_EQ(failing.steps, (std::vector<BigInt>{0, 1, 2}));
    EXPECT_LT(fast.steps.size(), 100u) << "stopped within a few steps";
}
