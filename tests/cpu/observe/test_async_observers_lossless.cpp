/// @file tests/cpu/observe/test_async_observers_lossless.cpp
/// @brief Online analysis (RDF / MSD / VACF) through the driver's shared,
///        lossless capture buffer.
///
/// Oracles independent of the code under test: the sample count is the
/// number of cadence steps written out by hand (initial state + every k-th
/// step), and a run whose buffer holds one frame behind a slow consumer must
/// produce bit-for-bit the curves of a run whose buffer never fills — a
/// skipped or reordered sample would change them.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "atomiverse/cpu/asynchronous/async_observer.h"
#include "atomiverse/cpu/asynchronous/capture_buffer.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/msd.h"
#include "atomiverse/cpu/observe/rdf.h"
#include "atomiverse/cpu/observe/vacf.h"
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

/// x by tag of one state.
using Positions = std::map<std::int64_t, double>;

Positions positions_after(int steps) {
    cpu::Frame frame = make_frame();
    cpu::md::MD sim;
    mount_lj(sim);
    sim.build();
    sim.run(frame, steps);
    Positions p;
    for (int i = 0; i < frame.nlocal(); ++i)
        p[frame.span(kAtom_tag)[i]] = frame.span(kAtom_x)[i];
    return p;
}

/// @brief A recorder-like consumer: keeps (step, x by tag), optionally slow.
class Probe : public cpu::md::AsyncObserver {
  public:
    Probe(std::string name, int every, std::chrono::milliseconds delay = 0ms)
        : name_(std::move(name)), every_(every), delay_(delay) {}
    std::string name() const override { return name_; }
    int frequency() const override { return every_; }
    CaptureColumns capture_columns() const override {
        CaptureColumns c;
        c.positions = c.velocities = c.tags = true;
        return c;
    }
    void on_snapshot(const cpu::FrameSnapshot& snap) override {
        if (delay_.count() > 0)
            std::this_thread::sleep_for(delay_);
        Positions p;
        for (int i = 0; i < snap.n_owned; ++i)
            p[snap.tag[i]] = snap.x[i];
        std::lock_guard<std::mutex> lk(mtx);
        steps.push_back(snap.step);
        frames.push_back(std::move(p));
    }
    std::mutex mtx;
    std::vector<BigInt> steps;
    std::vector<Positions> frames;

  private:
    std::string name_;
    int every_;
    std::chrono::milliseconds delay_;
};

/// @brief A temp path, any file already there removed.
std::string tmp(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove(p, ec);
    return p.string();
}

/// Curves of one run of RDF + MSD + VACF at every step.
struct Curves {
    std::vector<double> gr, msd, vacf;
    int rdf_samples = 0, msd_frames = 0, vacf_frames = 0;
    CaptureStats stats;
};

/// @param budget_frames 0: the default budget; else that many frames.
/// @param slow_ms       delay of an extra slow consumer (0: none).
Curves analysis_run(int n, std::uint64_t budget_frames, int slow_ms) {
    RDF rdf{5.5, 40, 1, tmp("atv_lossless_rdf.dat")};
    MSD msd{kDt, 1, tmp("atv_lossless_msd.dat"), /*window=*/4};
    VACF vacf{kDt, /*resolution=*/4, 1, tmp("atv_lossless_vacf.dat")};
    Probe slow{"slow", 1, std::chrono::milliseconds(slow_ms)};
    Curves out;
    {
        cpu::Frame f = make_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_observer_ref(rdf);
        sim.set_observer_ref(msd);
        sim.set_observer_ref(vacf);
        CaptureColumns columns = rdf.capture_columns() | msd.capture_columns()
                                 | vacf.capture_columns();
        if (slow_ms > 0) {
            sim.set_observer_ref(slow);
            columns = columns | slow.capture_columns();
        }
        if (budget_frames > 0)
            sim.set_capture_buffer_bytes(
                budget_frames * cpu::CaptureBuffer::frame_bytes(kN, columns));
        sim.build();
        sim.run(f, n);
        out.stats = sim.capture_stats();
    } // destroy → finalize
    out.gr = rdf.gr();
    out.msd = msd.curve();
    out.vacf = vacf.curve();
    out.rdf_samples = rdf.n_samples();
    out.msd_frames = msd.n_frames();
    out.vacf_frames = vacf.n_frames();
    for (const char* f : {"atv_lossless_rdf.dat", "atv_lossless_msd.dat",
                          "atv_lossless_vacf.dat"})
        tmp(f); // removes the file
    return out;
}

} // namespace

// Two recorder-like consumers and one analysis observer on the same cadence
// share ONE capture per step: a one-frame budget suffices, the buffer
// accounts one frame per captured step (not three), and each consumer sees
// the exact state of an independent run of that many steps.
TEST(CpuAsyncObserversLossless, SharedCaptureTakesOneFramePerStep) {
    constexpr int n = 8, k = 2;
    Probe first{"first", k}, second{"second", k};
    RDF rdf{5.5, 40, k, tmp("atv_shared_rdf.dat")};
    const CaptureColumns columns = first.capture_columns()
                                   | second.capture_columns()
                                   | rdf.capture_columns();
    const std::uint64_t frame = cpu::CaptureBuffer::frame_bytes(kN, columns);
    CaptureStats st;
    {
        cpu::Frame f = make_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_observer_ref(first);
        sim.set_observer_ref(second);
        sim.set_observer_ref(rdf);
        sim.set_capture_buffer_bytes(frame);
        sim.build();
        sim.run(f, n);
        st = sim.capture_stats();
    }
    tmp("atv_shared_rdf.dat"); // removes the file
    const std::vector<BigInt> want{0, 2, 4, 6, 8};
    EXPECT_EQ(first.steps, want);
    EXPECT_EQ(second.steps, want);
    EXPECT_EQ(rdf.n_samples(), 5);
    EXPECT_EQ(st.frame_bytes, frame);
    EXPECT_EQ(st.capacity, 1u);
    EXPECT_EQ(st.captures, want.size()) << "one capture per step, not three";
    EXPECT_EQ(st.peak_bytes, frame) << "never more than one frame in flight";
    for (std::size_t i = 0; i < want.size(); ++i) {
        const Positions oracle = positions_after(static_cast<int>(want[i]));
        for (const auto& [tag, x] : oracle) {
            EXPECT_NEAR(first.frames[i].at(tag), x, 1e-12);
            EXPECT_NEAR(second.frames[i].at(tag), x, 1e-12);
        }
    }
}

// RDF / MSD / VACF behind a one-frame budget and a slow consumer give the
// identical curves (bit for bit) and the same sample counts as a run whose
// buffer never fills: no sample is skipped or reordered. The counts are the
// cadence steps 0..n (n + 1 at every step).
TEST(CpuAsyncObserversLossless, CurvesIdenticalUnderBackpressure) {
    constexpr int n = 24;
    const Curves free_run = analysis_run(n, 0, 0);
    const Curves tight = analysis_run(n, 1, 3);
    EXPECT_EQ(free_run.stats.blocked_episodes, 0u);
    EXPECT_GT(tight.stats.blocked_episodes, 0u);
    EXPECT_EQ(tight.stats.capacity, 1u);
    for (const Curves* c : {&free_run, &tight}) {
        EXPECT_EQ(c->rdf_samples, n + 1);
        EXPECT_EQ(c->msd_frames, n + 1);
        EXPECT_EQ(c->vacf_frames, n + 1);
        EXPECT_EQ(c->stats.captures, static_cast<std::uint64_t>(n + 1));
    }
    ASSERT_FALSE(free_run.gr.empty());
    ASSERT_FALSE(free_run.msd.empty());
    ASSERT_FALSE(free_run.vacf.empty());
    EXPECT_EQ(tight.gr, free_run.gr);
    EXPECT_EQ(tight.msd, free_run.msd);
    EXPECT_EQ(tight.vacf, free_run.vacf);
}
