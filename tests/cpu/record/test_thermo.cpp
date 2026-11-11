/// @file tests/cpu/record/test_thermo.cpp
/// @brief Unit tests for `cpu::Thermo` — the LAMMPS `thermo` step convention.
///
/// LAMMPS increments `update->ntimestep` at the top of every step
/// (src/verlet.cpp Verlet::run). `Output::setup` (src/output.cpp) prints the
/// header and the current timestep at the start of every run; `Output::write`
/// prints at the end of a step whose timestep is a multiple of `thermo_every`
/// and on the run's last step. So a line labelled `s` is the state after `s`
/// steps. The oracle is not the recorder: each printed potential and kinetic
/// energy is compared with an independent driver that ran exactly `s` steps.
/// The lines are captured from the engine logger through a temporary
/// spdlog sink. Serial, sub-second, no files.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/sinks/ostream_sink.h>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/thermo.h"

using namespace ATV;

namespace {

constexpr int kN = 10;          // atoms
constexpr double kEps = 0.0104; // [eV]
constexpr double kSigma = 3.4;  // [Å]
constexpr double kCutoff = 6.0; // [Å]
constexpr double kSkin = 0.3;   // [Å]
constexpr double kDt = 1.0;     // [fs]

/// @brief Ten argon atoms 3.5 Å apart (repulsive, so energies move every
///        step) with a deterministic velocity field; no RNG anywhere.
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
        frame.span(kAtom_atomic_number)[i] = 18;
    }
    frame.box = cpu::Box::cubic(30.0);
    return frame;
}

void mount_lj(cpu::md::MD& sim) {
    sim.on(md::Phase::force).add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
}

/// @brief Oracle energies after exactly `steps` steps of a fresh driver
///        with no thermo: {pe, ke} [eV].
std::pair<double, double> energies_after(int steps) {
    cpu::Frame frame = make_frame();
    cpu::md::MD sim;
    mount_lj(sim);
    sim.build();
    sim.run(frame, steps);
    return {frame.meta.get(kEnergy), frame.meta.get(kKinetic)};
}

/// @brief Captures what the engine logger prints while alive.
class LogCapture {
  public:
    LogCapture()
        : sink_(std::make_shared<spdlog::sinks::ostream_sink_mt>(out_)) {
        // The first Runtime would call log_init and rebuild the sinks,
        // dropping ours: configure the default logging first.
        if (!ATV::log_initialized())
            ATV::log_init(ATV::LogConfig{});
        sink_->set_pattern("%v");
        ATV::logger().sinks().push_back(sink_);
    }
    ~LogCapture() {
        auto& sinks = ATV::logger().sinks();
        for (auto it = sinks.begin(); it != sinks.end(); ++it)
            if (*it == sink_) {
                sinks.erase(it);
                break;
            }
    }
    std::string text() {
        ATV::logger().flush();
        return out_.str();
    }

  private:
    std::ostringstream out_;
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink_;
};

struct Line {
    long long step;
    double pe, ke;
};

/// @brief Parse `step pe ke` thermo lines; counts the `Step` headers.
std::vector<Line> parse(const std::string& text, int& headers) {
    std::vector<Line> lines;
    headers = 0;
    std::istringstream in(text);
    std::string row;
    while (std::getline(in, row)) {
        std::istringstream ss(row);
        std::string first;
        if (!(ss >> first))
            continue;
        if (first == "Step") {
            ++headers;
            continue;
        }
        Line l{};
        std::istringstream num(first);
        if (!(num >> l.step) || !(ss >> l.pe >> l.ke))
            continue; // another log line
        lines.push_back(l);
    }
    return lines;
}

/// @brief Run with thermo `freq` over the chunks `runs`, capturing lines.
std::vector<Line> thermo_lines(int freq, const std::vector<int>& runs,
                               int& headers) {
    LogCapture capture;
    {
        cpu::Frame frame = make_frame();
        cpu::md::MD sim;
        mount_lj(sim);
        sim.set_thermo(freq, {"step", "pe", "ke"});
        sim.build();
        for (int n : runs)
            sim.run(frame, n);
    }
    return parse(capture.text(), headers);
}

/// @brief Each line is the state after `line.step` steps: the printed
///        energies (8 significant digits) equal the oracle's.
void expect_lines_are_oracle_states(const std::vector<Line>& lines) {
    for (const auto& l : lines) {
        SCOPED_TRACE("step " + std::to_string(l.step));
        const auto [pe, ke] = energies_after(static_cast<int>(l.step));
        EXPECT_NEAR(l.pe, pe, 1e-7 * std::abs(pe) + 1e-14);
        EXPECT_NEAR(l.ke, ke, 1e-7 * std::abs(ke) + 1e-14);
    }
}

} // namespace

// The oracle separates consecutive steps at the printed precision, so the
// checks below pin which step each line reports.
TEST(CpuThermoTest, OracleSeparatesConsecutiveSteps) {
    for (int s = 0; s < 6; ++s) {
        const auto a = energies_after(s), b = energies_after(s + 1);
        EXPECT_GT(std::abs(a.second - b.second), 1e-5 * std::abs(a.second))
            << "ke at steps " << s << " and " << s + 1;
    }
}

// A fresh 6-step run at freq 2: header, then steps 0, 2, 4, 6.
TEST(CpuThermoTest, FreshRunPrintsZeroThroughNEachTheStateAfterThatManySteps) {
    int headers = 0;
    const auto lines = thermo_lines(2, {6}, headers);
    std::vector<long long> steps;
    for (const auto& l : lines)
        steps.push_back(l.step);
    EXPECT_EQ(steps, (std::vector<long long>{0, 2, 4, 6}));
    EXPECT_EQ(headers, 1);
    expect_lines_are_oracle_states(lines);
}

// LAMMPS prints the last step of a run whatever the cadence
// (`next_thermo = MIN(next_thermo, update->laststep)`).
TEST(CpuThermoTest, LastStepIsPrintedOffCadence) {
    int headers = 0;
    const auto lines = thermo_lines(2, {5}, headers);
    std::vector<long long> steps;
    for (const auto& l : lines)
        steps.push_back(l.step);
    EXPECT_EQ(steps, (std::vector<long long>{0, 2, 4, 5}));
    expect_lines_are_oracle_states(lines);
}

// Each continued run opens with LAMMPS's setup output: a new header and the
// current state (step 4 again — LAMMPS `thermo->compute(0)` runs on every
// setup), then the loop continues the count.
TEST(CpuThermoTest, ContinuedRunOpensWithHeaderAndCurrentState) {
    int headers = 0;
    const auto lines = thermo_lines(2, {4, 4}, headers);
    std::vector<long long> steps;
    for (const auto& l : lines)
        steps.push_back(l.step);
    EXPECT_EQ(steps, (std::vector<long long>{0, 2, 4, 4, 6, 8}));
    EXPECT_EQ(headers, 2);
    expect_lines_are_oracle_states(lines);
}

// LAMMPS `thermo 0` (src/output.cpp: `next_thermo = update->laststep` when
// `thermo_every` is 0): the first and the last step of every run only.
TEST(CpuThermoTest, CadenceZeroPrintsFirstAndLastStepOfEachRun) {
    int headers = 0;
    auto lines = thermo_lines(0, {6}, headers);
    std::vector<long long> steps;
    for (const auto& l : lines)
        steps.push_back(l.step);
    EXPECT_EQ(steps, (std::vector<long long>{0, 6}));
    expect_lines_are_oracle_states(lines);

    lines = thermo_lines(0, {4, 4}, headers);
    steps.clear();
    for (const auto& l : lines)
        steps.push_back(l.step);
    EXPECT_EQ(steps, (std::vector<long long>{0, 4, 4, 8}));
    EXPECT_EQ(headers, 2);
    expect_lines_are_oracle_states(lines);
}
