/// @file tests/cpu/minimize/test_minimize_output.cpp
/// @brief The LAMMPS `minimize` output step convention of the minimizer-loop
///        writers: `TrajectoryWriter`, `OptLog`, `Thermo`, and the MD
///        recorders (`XyzRecorder`, `MrecRecorder`) bound in a minimization.
///
/// LAMMPS (src/min.cpp, src/min_cg.cpp, src/output.cpp, e8c4c278a8):
/// `Min::setup` ends with `Output::setup`, which dumps the initial state as
/// step 0 (and always prints thermo); each iteration starts with
/// `ntimestep = ++update->ntimestep`, so the output at the end of iteration
/// i is labelled with the completed-iteration count; thermo also prints the
/// budget's last iteration (`laststep`); a stop before MAXITER forces dumps
/// and thermo of the final state in `Min::run`; `last_dump` / `last_thermo`
/// never write a step twice. The oracle is never the writer: each record is
/// compared with an independent minimization that ran exactly that many
/// iterations. Deterministic LJ (no RNG), serial, tmp files only.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <spdlog/sinks/ostream_sink.h>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/minimize/opt_log.h"
#include "atomiverse/cpu/minimize/trajectory_writer.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/thermo.h"
#include "atomiverse/cpu/record/xyz_recorder.h"
#ifdef ATV_HAS_IO
#include "atomiverse/cpu/record/mrec_recorder.h"
#endif

using namespace ATV;

namespace {

constexpr int kN = 10;
constexpr double kEps = 0.0104; // [eV]
constexpr double kSigma = 3.4;  // [Å]
constexpr double kCutoff = 6.0; // [Å]
constexpr double kSkin = 0.3;   // [Å]

/// @brief Ten argon atoms on a compressed, slightly sheared 3.5 Å lattice:
///        far from the minimum, so every iteration moves the geometry.
cpu::Frame make_frame() {
    cpu::Frame frame(kN);
    for (int i = 0; i < kN; ++i) {
        frame.span(kAtom_x)[i] =
            8.0 + 3.5 * static_cast<double>(i % 5) + 0.05 * (i / 5);
        frame.span(kAtom_y)[i] = 10.0 + 3.5 * static_cast<double>(i / 5);
        frame.span(kAtom_z)[i] = 15.0 + 0.1 * static_cast<double>(i % 3);
        frame.span(kAtom_mass)[i] = 39.948;
        frame.span(kAtom_atomic_number)[i] = 18;
        frame.span(kAtom_tag)[i] = static_cast<TagInt>(i + 1);
    }
    frame.box = cpu::Box::cubic(30.0);
    return frame;
}

/// @brief A guard that stops the loop after `n` iterations (LAMMPS: any
///        stop condition other than MAXITER, e.g. `etol`).
class StopAfter : public cpu::Criterion {
  public:
    explicit StopAfter(BigInt n) : n_(n) {}
    std::string name() const override { return "stop_after"; }
    void launch(const cpu::LaunchContext&) override {}
    Verdict verdict(const cpu::LoopState& s) override {
        return s.iteration + 1 >= n_ ? Verdict::Stop : Verdict::Continue;
    }

  private:
    BigInt n_;
};

/// @brief LBFGS on LJ; convergence unreachable (tolerances 1e-30), so only
///        the budget or a `StopAfter` ends the loop.
void mount(cpu::minimizer::Minimize& opt) {
    opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    opt.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
    opt.add_criterion(cpu::ForceConverged{1e-30, 1e-30, 1e-30, 1e-30, 1e-30});
}

/// @brief Oracle: the fixture after exactly `iters` iterations of a fresh
///        minimization with no writer.
cpu::Frame state_after(int iters) {
    cpu::Frame frame = make_frame();
    if (iters == 0)
        return frame;
    cpu::minimizer::Minimize opt;
    mount(opt);
    opt.build();
    EXPECT_EQ(opt.run(frame, iters), iters);
    return frame;
}

/// @brief Oracle energy after exactly `iters` iterations [eV].
double energy_after(int iters) {
    cpu::Frame frame = make_frame();
    cpu::minimizer::Minimize opt;
    mount(opt);
    opt.build();
    opt.run(frame, iters); // iters = 0: the setup evaluation only
    return frame.meta.get(kEnergy);
}

std::string fresh_tmp(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    return p.string();
}

struct XyzFrame {
    long long step = -1;
    std::vector<std::array<double, 3>> pos;
};

/// @brief Frames of an extended-XYZ file with their `step=` comment meta.
std::vector<XyzFrame> read_xyz(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<XyzFrame> frames;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        const int n = std::stoi(line);
        XyzFrame f;
        std::getline(in, line);
        const auto at = line.find("step=");
        if (at == std::string::npos)
            throw std::runtime_error("no step= in " + line);
        f.step = std::stoll(line.substr(at + 5));
        for (int i = 0; i < n; ++i) {
            std::getline(in, line);
            std::istringstream ss(line);
            std::string sym;
            std::array<double, 3> r{};
            ss >> sym >> r[0] >> r[1] >> r[2];
            f.pos.push_back(r);
        }
        frames.push_back(std::move(f));
    }
    return frames;
}

/// @brief Every written atom sits on a distinct oracle atom.
void expect_positions_are(const std::vector<std::array<double, 3>>& pos,
                          const cpu::Frame& ref) {
    ASSERT_EQ(pos.size(), static_cast<std::size_t>(kN));
    std::vector<bool> used(kN, false);
    for (int i = 0; i < kN; ++i) {
        int best = -1;
        double best_d = 1e300;
        for (int j = 0; j < kN; ++j) {
            const auto& p = pos[static_cast<std::size_t>(i)];
            const double d = std::abs(p[0] - ref.span(kAtom_x)[j])
                             + std::abs(p[1] - ref.span(kAtom_y)[j])
                             + std::abs(p[2] - ref.span(kAtom_z)[j]);
            if (d < best_d) {
                best_d = d;
                best = j;
            }
        }
        ASSERT_GE(best, 0);
        EXPECT_FALSE(used[static_cast<std::size_t>(best)]);
        used[static_cast<std::size_t>(best)] = true;
        EXPECT_LT(best_d, 1e-9) << "atom " << i << " is not the oracle state";
    }
}

std::vector<long long> steps_of(const std::vector<XyzFrame>& frames) {
    std::vector<long long> s;
    for (const auto& f : frames)
        s.push_back(f.step);
    return s;
}

/// @brief Captures the engine logger while alive (see test_thermo.cpp).
class LogCapture {
  public:
    LogCapture()
        : sink_(std::make_shared<spdlog::sinks::ostream_sink_mt>(out_)) {
        if (!ATV::log_initialized())
            ATV::log_init(ATV::LogConfig{});
        sink_->set_pattern("%v");
        ATV::logger().sinks().push_back(sink_);
    }
    ~LogCapture() {
        auto& sinks = ATV::logger().sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_),
                    sinks.end());
    }
    std::string text() {
        ATV::logger().flush();
        return out_.str();
    }

  private:
    std::ostringstream out_;
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink_;
};

/// @brief `{step, pe}` rows of `Step PotEng` thermo lines.
std::vector<std::pair<long long, double>> thermo_rows(const std::string& text) {
    std::vector<std::pair<long long, double>> rows;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        long long step = 0;
        double pe = 0;
        std::string rest;
        if ((ss >> step >> pe) && !(ss >> rest))
            rows.emplace_back(step, pe);
    }
    return rows;
}

} // namespace

// The oracle separates consecutive iterations far beyond the tolerances.
TEST(MinimizeOutput, OracleSeparatesConsecutiveIterations) {
    for (int i = 0; i < 7; ++i) {
        const cpu::Frame a = state_after(i), b = state_after(i + 1);
        double d = 0.0;
        for (int j = 0; j < kN; ++j)
            d = std::max(
                d, std::abs(a.span(kAtom_x)[j] - b.span(kAtom_x)[j])
                       + std::abs(a.span(kAtom_y)[j] - b.span(kAtom_y)[j]));
        EXPECT_GT(d, 1e-6) << "iterations " << i << " and " << i + 1;
        const double ea = energy_after(i), eb = energy_after(i + 1);
        EXPECT_GT(std::abs(ea - eb), 1e-4 * std::abs(ea))
            << "energies at iterations " << i << " and " << i + 1;
    }
}

// ── TrajectoryWriter ─────────────────────────────────────────────────────

// Stopped by a criterion after 7 iterations at stride 3: the initial
// geometry, iterations 3 and 6, and the final iteration 7 off cadence.
TEST(MinimizeOutput, TrajectoryWriterEarlyStopWritesTheFinalStateOffCadence) {
    const std::string path = fresh_tmp("atv_min_traj_stop.xyz");
    {
        cpu::Frame frame = make_frame();
        cpu::minimizer::Minimize opt;
        mount(opt);
        opt.add_criterion(StopAfter{7});
        opt.on(minimizer::Phase::end_of_step)
            .add(cpu::minimizer::TrajectoryWriter{path, 3});
        opt.build();
        EXPECT_EQ(opt.run(frame, 50), 7);
        const auto frames = read_xyz(path);
        EXPECT_EQ(steps_of(frames), (std::vector<long long>{0, 3, 6, 7}));
        for (const auto& f : frames) {
            SCOPED_TRACE("step " + std::to_string(f.step));
            expect_positions_are(f.pos, state_after(static_cast<int>(f.step)));
        }
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// The budget runs out (LAMMPS MAXITER): no forced final write, and a final
// iteration on cadence is written once.
TEST(MinimizeOutput, TrajectoryWriterBudgetExhaustionWritesOnCadenceOnly) {
    for (const auto& [stride, want] :
         {std::pair<int, std::vector<long long>>{4, {0, 4}},
          std::pair<int, std::vector<long long>>{3, {0, 3, 6}}}) {
        SCOPED_TRACE("stride " + std::to_string(stride));
        const std::string path = fresh_tmp("atv_min_traj_budget.xyz");
        cpu::Frame frame = make_frame();
        cpu::minimizer::Minimize opt;
        mount(opt);
        opt.on(minimizer::Phase::end_of_step)
            .add(cpu::minimizer::TrajectoryWriter{path, stride});
        opt.build();
        EXPECT_EQ(opt.run(frame, 6), 6);
        const auto frames = read_xyz(path);
        EXPECT_EQ(steps_of(frames), want);
        for (const auto& f : frames)
            expect_positions_are(f.pos, state_after(static_cast<int>(f.step)));
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
}

// ── XyzRecorder bound in a minimization ──────────────────────────────────

// The MD recorder follows the same rules on the minimizer loop.
TEST(MinimizeOutput, XyzRecorderEarlyStopWritesTheFinalState) {
    const std::string path = fresh_tmp("atv_min_xyzrec_stop.xyz");
    {
        cpu::Frame frame = make_frame();
        cpu::minimizer::Minimize opt;
        mount(opt);
        opt.add_criterion(StopAfter{5});
        opt.on(minimizer::Phase::end_of_step)
            .add(cpu::XyzRecorder{path, 2}, +10);
        opt.build();
        opt.run(frame, 50);
    }
    const auto frames = read_xyz(path);
    EXPECT_EQ(steps_of(frames), (std::vector<long long>{0, 2, 4, 5}));
    for (const auto& f : frames)
        expect_positions_are(f.pos, state_after(static_cast<int>(f.step)));
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── OptLog ───────────────────────────────────────────────────────────────

// Row 0 is the initial state; row s the state after s iterations; the
// early stop after 5 ends on row 5 without repeating it.
TEST(MinimizeOutput, OptLogRowsAreTheStateAfterTheirIteration) {
    const std::string path = fresh_tmp("atv_min_optlog.tsv");
    {
        cpu::Frame frame = make_frame();
        cpu::minimizer::Minimize opt;
        mount(opt);
        opt.add_criterion(StopAfter{5});
        opt.on(minimizer::Phase::end_of_step).add(cpu::minimizer::OptLog{path});
        opt.build();
        opt.run(frame, 50);
    }
    std::ifstream in(path);
    std::string line;
    std::getline(in, line); // header
    std::vector<long long> steps;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        long long step = 0;
        double e = 0, fmax = 0, frms = 0, dx = 0;
        ASSERT_TRUE(static_cast<bool>(ss >> step >> e >> fmax >> frms >> dx))
            << line;
        steps.push_back(step);
        const double want = energy_after(static_cast<int>(step));
        EXPECT_NEAR(e, want, 1e-5 * std::abs(want)) << "step " << step;
        if (step == 0)
            EXPECT_EQ(dx, 0.0);
    }
    EXPECT_EQ(steps, (std::vector<long long>{0, 1, 2, 3, 4, 5}));
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── Thermo ───────────────────────────────────────────────────────────────

// Thermo 3 stopped after 7: steps 0, 3, 6 and the final 7 (LAMMPS Min::run
// forces the last line); thermo 0 over a 6-iteration budget: the first and
// the last step only (LAMMPS `thermo 0`).
TEST(MinimizeOutput, ThermoPrintsSetupCadenceAndFinalStep) {
    struct Case {
        int every;
        int budget;
        BigInt stop_after; // 0: none
        std::vector<long long> want;
    };
    for (const Case& c : {Case{3, 50, 7, {0, 3, 6, 7}}, Case{0, 6, 0, {0, 6}},
                          Case{2, 6, 0, {0, 2, 4, 6}}}) {
        SCOPED_TRACE("every " + std::to_string(c.every) + " budget "
                     + std::to_string(c.budget));
        LogCapture capture;
        {
            cpu::Frame frame = make_frame();
            cpu::minimizer::Minimize opt;
            mount(opt);
            if (c.stop_after > 0)
                opt.add_criterion(StopAfter{c.stop_after});
            opt.set_thermo(c.every, {"step", "pe"});
            opt.build();
            opt.run(frame, c.budget);
        }
        const auto rows = thermo_rows(capture.text());
        std::vector<long long> steps;
        for (const auto& [step, pe] : rows) {
            steps.push_back(step);
            const double want = energy_after(static_cast<int>(step));
            EXPECT_NEAR(pe, want, 1e-7 * std::abs(want)) << "step " << step;
        }
        EXPECT_EQ(steps, c.want);
    }
}

// ── Single pass (Evaluate, LAMMPS `run 0`) ──────────────────────────────

// The pass is its own setup output: the state is written as step 0 whatever
// the cadence, and thermo 0 still prints it.
TEST(MinimizeOutput, EvaluatePassWritesItsStateAsStepZero) {
    const std::string path = fresh_tmp("atv_eval_xyz.xyz");
    LogCapture capture;
    {
        cpu::Frame frame = make_frame();
        cpu::Evaluate eval;
        eval.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
        eval.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
        eval.on(ATV::evaluate::Phase::finalize)
            .add(cpu::XyzRecorder{path, /*freq=*/3}, +10);
        eval.set_thermo(0, {"step", "pe"});
        eval.build();
        eval.run(frame);
    }
    const auto frames = read_xyz(path);
    ASSERT_EQ(frames.size(), 1u);
    EXPECT_EQ(frames[0].step, 0);
    expect_positions_are(frames[0].pos, make_frame());
    const auto rows = thermo_rows(capture.text());
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].first, 0);
    EXPECT_NEAR(rows[0].second, energy_after(0),
                1e-7 * std::abs(energy_after(0)));
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

#ifdef ATV_HAS_IO
// ── MrecRecorder bound in a minimization ─────────────────────────────────

namespace {

class StepLog : public cpu::MrecRecorder {
  public:
    using cpu::MrecRecorder::MrecRecorder;
    std::vector<std::int64_t> steps;

  protected:
    void append_frame(const cpu::Frame& frame, std::int64_t step, double time,
                      bool has_time) override {
        steps.push_back(step);
        cpu::MrecRecorder::append_frame(frame, step, time, has_time);
    }
};

} // namespace

// No integrator: steps only; the early stop's final state lands once.
TEST(MinimizeOutput, MrecRecorderEarlyStopWritesTheFinalStateOnce) {
    const std::string path = fresh_tmp("atv_min_mrec_stop.mrec");
    StepLog rec{path, 3, 0, false};
    {
        cpu::Frame frame = make_frame();
        cpu::minimizer::Minimize opt;
        mount(opt);
        opt.add_criterion(StopAfter{6});
        opt.on(minimizer::Phase::end_of_step).add_ref(rec, +10);
        opt.build();
        opt.run(frame, 50);
    }
    rec.close();
    // Step 6 is on cadence and written by the loop; the forced final write
    // of step 6 is skipped (LAMMPS `last_dump`).
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 3, 6}));
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}
#endif
