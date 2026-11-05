/// @file tests/cuda/minimize/test_minimize_output.cu
/// @brief The LAMMPS `minimize` output step convention on the CUDA
///        minimizer loop: `cuda::MrecRecorder`, `cuda::XyzRecorder`,
///        `cuda::Thermo`.
///
/// LAMMPS (src/min.cpp, src/min_cg.cpp, src/output.cpp, e8c4c278a8):
/// `Min::setup` → `Output::setup` writes the initial state as step 0; each
/// iteration starts with `++update->ntimestep`, so the output at the end of
/// iteration i carries the completed-iteration count; thermo also prints the
/// budget's last iteration; a stop before MAXITER forces dumps and thermo of
/// the final state (`Min::run`); `last_dump` / `last_thermo` write no step
/// twice. The oracle is never the recorder: each record is compared with an
/// independent CUDA minimization of exactly that many iterations.
/// Deterministic LJ, tmp files only.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/criterion.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/record/recorder.cuh"
#include "atomiverse/cuda/record/thermo.cuh"
#ifdef ATV_HAS_IO
#include "atomiverse/cuda/record/mrec_recorder.cuh"
#include "atomiverse/molpy.hpp"
#endif

using namespace ATV;

namespace {

constexpr int kN = 10;
constexpr double kEps = 0.0104;                             // [eV]
constexpr double kSigma = 3.4;                              // [Å]
constexpr double kCutoff = 6.0;                             // [Å]
constexpr double kSkin = 0.3;                               // [Å]
const double kTol = sizeof(cuda::pos_t) == 8 ? 1e-8 : 1e-4; // [Å]

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
        frame.span(kAtom_type)[i] = 18;
        frame.span(kAtom_atomic_number)[i] = 18;
        frame.span(kAtom_tag)[i] = static_cast<TagInt>(i + 1);
    }
    frame.box = cpu::Box::cubic(30.0);
    return frame;
}

/// @brief Stop after `n` iterations (any LAMMPS stop condition but MAXITER).
class StopAfter : public cuda::Criterion {
  public:
    explicit StopAfter(BigInt n) : n_(n) {}
    std::string name() const override { return "stop_after"; }
    Verdict verdict(const cuda::LoopState& s) override {
        return s.iteration + 1 >= n_ ? Verdict::Stop : Verdict::Continue;
    }

  private:
    BigInt n_;
};

/// @brief LBFGS on LJ; convergence unreachable, so only the budget or a
///        `StopAfter` ends the loop.
void mount(cuda::minimizer::Minimize& min) {
    min.set_optimizer(cuda::minimizer::LBFGS{5, 0.01, 10});
    min.on(minimizer::Phase::force)
        .add(cuda::NeighborList{kCutoff, kSkin}, -10)
        .add(cuda::PairLJCut{kEps, kSigma, kCutoff});
    min.add_criterion(cuda::ForceConverged{1e-30, 1e-30, 1e-30, 1e-30, 1e-30});
}

/// @brief Oracle: the fixture after exactly `iters` iterations of a fresh
///        CUDA minimization with no recorder (`iters = 0`: setup only).
cpu::Frame state_after(int iters) {
    cpu::Frame frame = make_frame();
    cuda::minimizer::Minimize min;
    mount(min);
    min.build();
    EXPECT_EQ(min.run(frame, iters), iters);
    return frame;
}

std::string fresh_tmp(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(p, ec);
    return p.string();
}

void remove_path(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

/// @brief Every recorded atom sits on a distinct oracle atom.
void expect_positions_are(const std::vector<std::array<double, 3>>& pos,
                          const cpu::Frame& ref) {
    ASSERT_EQ(pos.size(), static_cast<std::size_t>(kN));
    std::vector<bool> used(kN, false);
    for (int i = 0; i < kN; ++i) {
        const auto& p = pos[static_cast<std::size_t>(i)];
        int best = -1;
        double best_d = 1e300;
        for (int j = 0; j < kN; ++j) {
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
        EXPECT_LT(best_d, kTol) << "atom " << i << " is not the oracle state";
    }
}

/// @brief Positions of every XYZ frame (count line, comment, `Z x y z`).
std::vector<std::vector<std::array<double, 3>>>
read_xyz(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<std::array<double, 3>>> frames;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        const int n = std::stoi(line);
        std::getline(in, line);
        std::vector<std::array<double, 3>> pos;
        for (int i = 0; i < n; ++i) {
            std::getline(in, line);
            std::istringstream ss(line);
            std::string sym;
            std::array<double, 3> r{};
            ss >> sym >> r[0] >> r[1] >> r[2];
            pos.push_back(r);
        }
        frames.push_back(std::move(pos));
    }
    return frames;
}

/// @brief `{step, pe}` rows of a thermo file.
std::vector<std::pair<long long, double>> thermo_rows(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::pair<long long, double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        double step = 0, pe = 0;
        if (ss >> step >> pe)
            rows.emplace_back(static_cast<long long>(step), pe);
    }
    return rows;
}

std::vector<std::array<double, 3>> positions(const cpu::Frame& f) {
    std::vector<std::array<double, 3>> r;
    for (int i = 0; i < f.nlocal(); ++i)
        r.push_back(
            {f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]});
    return r;
}

} // namespace

// The oracle separates consecutive iterations far beyond the tolerances.
TEST(CudaMinimizeOutput, OracleSeparatesConsecutiveIterations) {
    for (int i = 0; i < 7; ++i) {
        const cpu::Frame a = state_after(i), b = state_after(i + 1);
        double d = 0.0;
        for (int j = 0; j < kN; ++j)
            d = std::max(
                d, std::abs(a.span(kAtom_x)[j] - b.span(kAtom_x)[j])
                       + std::abs(a.span(kAtom_y)[j] - b.span(kAtom_y)[j]));
        EXPECT_GT(d, 100 * kTol) << "iterations " << i << " and " << i + 1;
        const double ea = a.meta.get(kEnergy), eb = b.meta.get(kEnergy);
        EXPECT_GT(std::abs(ea - eb), 1e-4 * std::abs(ea))
            << "energies at iterations " << i << " and " << i + 1;
    }
    // The oracle's own setup state is the fixture.
    expect_positions_are(positions(state_after(0)), make_frame());
}

// Stopped after 5 iterations at every 2: the initial state, iterations 2
// and 4, and the final 5 off cadence.
TEST(CudaMinimizeOutput, XyzEarlyStopWritesTheFinalStateOffCadence) {
    const std::string path = fresh_tmp("atv_cuda_min_xyz.xyz");
    {
        cpu::Frame frame = make_frame();
        cuda::minimizer::Minimize min;
        mount(min);
        min.add_criterion(StopAfter{5});
        min.on(minimizer::Phase::end_of_step)
            .add(cuda::XyzRecorder{path, 2}, +10);
        min.build();
        EXPECT_EQ(min.run(frame, 50), 5);
    } // teardown drains the callbacks
    const auto frames = read_xyz(path);
    const int want[] = {0, 2, 4, 5};
    ASSERT_EQ(frames.size(), 4u);
    for (std::size_t f = 0; f < frames.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        expect_positions_are(frames[f], state_after(want[f]));
    }
    remove_path(path);
}

// Thermo every 3 stopped after 7: 0, 3, 6 and the final 7. Thermo every 0
// over a 6-iteration budget: the first and the last iteration (LAMMPS
// `thermo 0`, the budget's last iteration is `laststep`). Thermo every 2
// over 6: 0, 2, 4, 6 with 6 printed once.
TEST(CudaMinimizeOutput, ThermoPrintsSetupCadenceAndFinalIteration) {
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
        const std::string path = fresh_tmp("atv_cuda_min_thermo.dat");
        {
            cpu::Frame frame = make_frame();
            cuda::minimizer::Minimize min;
            mount(min);
            if (c.stop_after > 0)
                min.add_criterion(StopAfter{c.stop_after});
            min.on(minimizer::Phase::end_of_step)
                .add(cuda::Thermo{c.every, {"step", "pe"}, path}, +20);
            min.build();
            min.run(frame, c.budget);
        }
        std::vector<long long> steps;
        for (const auto& [step, pe] : thermo_rows(path)) {
            steps.push_back(step);
            const double want =
                state_after(static_cast<int>(step)).meta.get(kEnergy);
            EXPECT_NEAR(pe, want, 1e-7 * std::abs(want) + 1e-12)
                << "step " << step;
        }
        EXPECT_EQ(steps, c.want);
        remove_path(path);
    }
}

#ifdef ATV_HAS_IO
namespace {

/// @brief Logs the step axis on the worker.
class StepLog : public cuda::MrecRecorder {
  public:
    StepLog(std::string path, int freq)
        : cuda::MrecRecorder(std::move(path), freq, 0, false) {}
    std::vector<std::int64_t> steps;

  protected:
    void append_frame(const cpu::Frame& frame, std::int64_t step, double time,
                      bool has_time) override {
        steps.push_back(step);
        cuda::MrecRecorder::append_frame(frame, step, time, has_time);
    }
};

/// @brief Store frame `index` holds `ref`'s positions, atom by atom (tag).
void expect_store_frame_is(const std::string& path, std::uint64_t index,
                           const cpu::Frame& ref) {
    auto fr =
        molrs::read_mrec_trajectory_frame({path.data(), path.size()}, index);
    auto tag = molrs::frame_column_i32(*fr, "atoms", "tag");
    ASSERT_EQ(tag.size(), static_cast<std::size_t>(kN));
    const auto* rtag = ref.span(kAtom_tag).data();
    const std::pair<const char*, const double*> cols[] = {
        {"x", ref.span(kAtom_x)},
        {"y", ref.span(kAtom_y)},
        {"z", ref.span(kAtom_z)}};
    for (const auto& [name, want] : cols) {
        auto got = molrs::frame_column_f64(*fr, "atoms", name);
        for (std::size_t i = 0; i < tag.size(); ++i) {
            int r = -1;
            for (int j = 0; j < kN; ++j)
                if (rtag[j] == tag[i])
                    r = j;
            ASSERT_GE(r, 0);
            EXPECT_NEAR(got[i], want[r], kTol) << name << " of tag " << tag[i];
        }
    }
}

} // namespace

// Stopped after 7 at every 3: 0, 3, 6 and the forced final 7.
TEST(CudaMinimizeOutput, MrecEarlyStopWritesTheFinalStateOffCadence) {
    const std::string path = fresh_tmp("atv_cuda_min_stop.mrec");
    StepLog rec{path, 3};
    {
        cpu::Frame frame = make_frame();
        cuda::minimizer::Minimize min;
        mount(min);
        min.add_criterion(StopAfter{7});
        min.on(minimizer::Phase::end_of_step).add_ref(rec, +10);
        min.build();
        EXPECT_EQ(min.run(frame, 50), 7);
        rec.close();
    }
    EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 3, 6, 7}));
    for (std::size_t f = 0; f < rec.steps.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        expect_store_frame_is(path, f,
                              state_after(static_cast<int>(rec.steps[f])));
    }
    remove_path(path);
}

// Stopped after 6 at every 3: the loop already wrote step 6, so the forced
// final write is skipped on the device (LAMMPS `last_dump`); a 6-iteration
// budget (MAXITER) at every 4 forces nothing: 0 and 4 only.
TEST(CudaMinimizeOutput, MrecWritesNoStepTwiceAndForcesNothingAtMaxiter) {
    {
        const std::string path = fresh_tmp("atv_cuda_min_once.mrec");
        StepLog rec{path, 3};
        {
            cpu::Frame frame = make_frame();
            cuda::minimizer::Minimize min;
            mount(min);
            min.add_criterion(StopAfter{6});
            min.on(minimizer::Phase::end_of_step).add_ref(rec, +10);
            min.build();
            min.run(frame, 50);
            rec.close();
        }
        EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 3, 6}));
        remove_path(path);
    }
    {
        const std::string path = fresh_tmp("atv_cuda_min_maxiter.mrec");
        StepLog rec{path, 4};
        {
            cpu::Frame frame = make_frame();
            cuda::minimizer::Minimize min;
            mount(min);
            min.on(minimizer::Phase::end_of_step).add_ref(rec, +10);
            min.build();
            EXPECT_EQ(min.run(frame, 6), 6);
            rec.close();
        }
        EXPECT_EQ(rec.steps, (std::vector<std::int64_t>{0, 4}));
        expect_store_frame_is(path, 1, state_after(4));
        remove_path(path);
    }
}
#endif
