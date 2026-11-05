/// @file tests/cpu/asynchronous/test_capture_buffer.cpp
/// @brief The CPU driver's shared, lossless capture buffer
///        (`cpu::CaptureBuffer` + `cpu::AsyncConsumer`).
///
/// Oracles are independent of the code under test: every delivered frame is
/// compared with a fresh MD run of exactly that many steps and no consumer;
/// the expected step list is written out by hand from the LAMMPS output
/// convention (initial state, then every k-th completed step, never twice).
/// Sub-second except the deliberately slow consumers (a few ms per frame).

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "atomiverse/cpu/asynchronous/async_observer.h"
#include "atomiverse/cpu/asynchronous/capture_buffer.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

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
        frame.span(kAtom_tag)[i] = static_cast<TagInt>(i + 1);
    }
    frame.box = cpu::Box::cubic(kBox);
    return frame;
}

void mount_lj(cpu::md::MD& sim) {
    sim.on(md::Phase::force).add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
}

/// Per-atom (by tag) x and vx of one state.
struct State {
    std::map<std::int64_t, double> x, vx;
};

/// @brief Oracle: the fixture after exactly `steps` steps of a fresh driver
///        with no consumer.
State state_after(int steps) {
    cpu::Frame frame = make_frame();
    cpu::md::MD sim;
    mount_lj(sim);
    sim.build();
    sim.run(frame, steps);
    State s;
    for (int i = 0; i < frame.nlocal(); ++i) {
        s.x[frame.span(kAtom_tag)[i]] = frame.span(kAtom_x)[i];
        s.vx[frame.span(kAtom_tag)[i]] = frame.span(kAtom_vx)[i];
    }
    return s;
}

/// @brief Test consumer: keeps every delivered (step, x, vx by tag);
///        optionally slow, optionally failing at one step.
class Probe : public cpu::md::AsyncObserver {
  public:
    Probe(std::string name, int every, std::chrono::milliseconds delay = 0ms,
          BigInt fail_at = -1)
        : name_(std::move(name)), every_(every), delay_(delay),
          fail_at_(fail_at) {}

    std::string name() const override { return name_; }
    int frequency() const override { return every_; }
    CaptureColumns capture_columns() const override {
        CaptureColumns c;
        c.positions = true;
        c.velocities = true;
        c.tags = true;
        return c;
    }
    void on_snapshot(const cpu::FrameSnapshot& snap) override {
        if (delay_.count() > 0)
            std::this_thread::sleep_for(delay_);
        if (snap.step == fail_at_)
            throw std::runtime_error("injected: cannot write 'probe.out'");
        State s;
        for (int i = 0; i < snap.n_owned; ++i) {
            s.x[snap.tag[i]] = snap.x[i];
            s.vx[snap.tag[i]] = snap.vx[i];
        }
        std::lock_guard<std::mutex> lk(mtx);
        steps.push_back(snap.step);
        states.push_back(std::move(s));
    }
    void finalize() override { ++finalized; }

    std::mutex mtx;
    std::vector<BigInt> steps;
    std::vector<State> states;
    int finalized = 0;

  private:
    std::string name_;
    int every_;
    std::chrono::milliseconds delay_;
    BigInt fail_at_;
};

/// @brief Every delivered frame is the state of an independent run of that
///        many steps.
void expect_oracle_states(Probe& probe) {
    ASSERT_EQ(probe.steps.size(), probe.states.size());
    for (std::size_t f = 0; f < probe.steps.size(); ++f) {
        SCOPED_TRACE(probe.name() + " step " + std::to_string(probe.steps[f]));
        const State want = state_after(static_cast<int>(probe.steps[f]));
        ASSERT_EQ(probe.states[f].x.size(), want.x.size());
        for (const auto& [tag, x] : want.x) {
            EXPECT_NEAR(probe.states[f].x.at(tag), x, 1e-12) << "tag " << tag;
            EXPECT_NEAR(probe.states[f].vx.at(tag), want.vx.at(tag), 1e-14)
                << "tag " << tag;
        }
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

} // namespace

// The oracle separates consecutive steps far beyond the tolerances above.
TEST(CpuCaptureBuffer, OracleSeparatesConsecutiveSteps) {
    const State a = state_after(2), b = state_after(3);
    double dx = 0.0;
    for (const auto& [tag, x] : a.x)
        dx = std::max(dx, std::abs(x - b.x.at(tag)));
    EXPECT_GT(dx, 1e-6);
}

// A 0 budget is refused by the driver knob itself.
TEST(CpuCaptureBuffer, ZeroBudgetIsRefused) {
    cpu::md::MD sim;
    EXPECT_THROW(sim.set_capture_buffer_bytes(0), std::invalid_argument);
    EXPECT_EQ(sim.capture_buffer_bytes(), kCaptureBufferBytes);
}

// A budget below one captured frame is refused when the run starts, naming
// both sizes; nothing runs.
TEST(CpuCaptureBuffer, BudgetBelowOneFrameIsRefused) {
    const std::uint64_t frame =
        cpu::CaptureBuffer::frame_bytes(kN, probe_columns());
    Probe probe{"probe", 1};
    cpu::Frame f = make_frame();
    cpu::md::MD sim;
    mount_lj(sim);
    sim.set_observer_ref(probe);
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

// A normal run with the default budget delivers every cadence step before
// run() returns, never blocks, and finalizes each consumer once.
TEST(CpuCaptureBuffer, NormalRunDrainsWithoutBlocking) {
    Probe probe{"probe", 2};
    {
        cpu::Frame f = make_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_observer_ref(probe);
        sim.build();
        sim.run(f, 6);
        // Drained before run() returned — no wait here.
        EXPECT_EQ(probe.steps, every_k(2, 6));
        const CaptureStats st = sim.capture_stats();
        EXPECT_EQ(st.budget_bytes, kCaptureBufferBytes);
        EXPECT_EQ(st.frame_bytes,
                  cpu::CaptureBuffer::frame_bytes(kN, probe_columns()));
        EXPECT_EQ(st.captures, 4u);
        EXPECT_EQ(st.blocked_episodes, 0u);
        EXPECT_EQ(st.blocked_seconds, 0.0);
        EXPECT_GE(st.peak_bytes, st.frame_bytes);
        EXPECT_EQ(probe.finalized, 0);
    }
    EXPECT_EQ(probe.finalized, 1);
    expect_oracle_states(probe);
}

// Continued runs extend one series: the boundary step is delivered once.
TEST(CpuCaptureBuffer, ContinuedRunDeliversTheBoundaryStepOnce) {
    Probe probe{"probe", 2};
    cpu::Frame f = make_frame();
    cpu::md::MD sim;
    mount_lj(sim);
    sim.set_observer_ref(probe);
    sim.build();
    sim.run(f, 4);
    sim.run(f, 4);
    EXPECT_EQ(probe.steps, every_k(2, 8));
    expect_oracle_states(probe);
}

namespace {

/// Shared slow-consumer scenario: a slow and a fast consumer on one buffer of
/// `frames` frames, every step of an n-step run.
void slow_consumer_loses_nothing(std::uint64_t frames) {
    constexpr int n = 12;
    Probe slow{"slow", 1, 4ms};
    Probe fast{"fast", 1};
    const std::uint64_t frame =
        cpu::CaptureBuffer::frame_bytes(kN, probe_columns());
    CaptureStats st;
    {
        cpu::Frame f = make_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_observer_ref(slow);
        sim.set_observer_ref(fast);
        sim.set_capture_buffer_bytes(frames * frame);
        sim.build();
        sim.run(f, n);
        st = sim.capture_stats();
    }
    EXPECT_EQ(slow.steps, every_k(1, n));
    EXPECT_EQ(fast.steps, every_k(1, n));
    EXPECT_EQ(st.capacity, frames);
    EXPECT_EQ(st.captures, static_cast<std::uint64_t>(n + 1))
        << "one capture per step, shared by both consumers";
    EXPECT_EQ(st.peak_bytes, frames * frame) << "the buffer filled";
    EXPECT_GT(st.blocked_episodes, 0u);
    EXPECT_GT(st.blocked_seconds, 0.0);
    expect_oracle_states(slow);
    expect_oracle_states(fast);
}

} // namespace

TEST(CpuCaptureBuffer, SlowConsumerBehindAOneFrameBudgetBlocksButLosesNothing) {
    slow_consumer_loses_nothing(1);
}

TEST(CpuCaptureBuffer, SlowConsumerBehindATwoFrameBudgetBlocksButLosesNothing) {
    slow_consumer_loses_nothing(2);
}

// Consumers on different cadences share one capture where they coincide and
// each receives exactly its own steps.
TEST(CpuCaptureBuffer, DifferentCadencesShareCoincidentCaptures) {
    Probe two{"two", 2}, three{"three", 3};
    CaptureStats st;
    {
        cpu::Frame f = make_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_observer_ref(two);
        sim.set_observer_ref(three);
        sim.build();
        sim.run(f, 12);
        st = sim.capture_stats();
    }
    EXPECT_EQ(two.steps, every_k(2, 12));
    EXPECT_EQ(three.steps, every_k(3, 12));
    // Union of {0,2,…,12} and {0,3,…,12}: 0 2 3 4 6 8 9 10 12.
    EXPECT_EQ(st.captures, 9u);
    expect_oracle_states(two);
    expect_oracle_states(three);
}

// A consumer that fails mid-run stops the run promptly with an error naming
// the consumer and the cause; the producer never waits on space that will
// not free (one-frame budget). Bounded by a watchdog.
TEST(CpuCaptureBuffer, ConsumerFailureStopsTheRunPromptly) {
    Probe failing{"failing_probe", 1, 1ms, /*fail_at=*/3};
    Probe fast{"fast", 1};
    const std::uint64_t frame =
        cpu::CaptureBuffer::frame_bytes(kN, probe_columns());
    auto run = std::async(std::launch::async, [&]() -> std::string {
        cpu::Frame f = make_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_observer_ref(failing);
        sim.set_observer_ref(fast);
        sim.set_capture_buffer_bytes(frame);
        sim.build();
        try {
            sim.run(f, 100000);
        } catch (const std::runtime_error& e) {
            return e.what();
        }
        return "";
    });
    ASSERT_EQ(run.wait_for(60s), std::future_status::ready)
        << "a consumer failure must stop the run, not hang it";
    const std::string what = run.get();
    ASSERT_FALSE(what.empty()) << "the run must fail";
    EXPECT_NE(what.find("cpu::failing_probe"), std::string::npos) << what;
    EXPECT_NE(what.find("cannot write 'probe.out'"), std::string::npos) << what;
    EXPECT_NE(what.find("step 3"), std::string::npos) << what;
    EXPECT_EQ(failing.steps, (std::vector<BigInt>{0, 1, 2}));
    // Stopped within a few steps of the failure, not after 100000.
    EXPECT_LT(fast.steps.size(), 100u);
    EXPECT_EQ(failing.finalized, 1);
}
