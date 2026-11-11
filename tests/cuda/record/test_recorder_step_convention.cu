/// @file tests/cuda/record/test_recorder_step_convention.cu
/// @brief The LAMMPS output step convention of the CUDA AsyncRecorders —
///        `cuda::XyzRecorder`, `cuda::Thermo`, `cuda::md::PressureSeries`.
///
/// LAMMPS increments `update->ntimestep` at the top of every step
/// (src/verlet.cpp Verlet::run). `Output::setup` (src/output.cpp) writes
/// dumps — and always thermo — for the current timestep at the start of
/// every run; `Output::write` writes at the end of a step whose timestep is
/// a multiple of the cadence (thermo also on the run's last step); a dump
/// never writes a timestep twice (`last_dump`). So a record labelled `s` is
/// the state after `s` steps. The oracle is never the recorder: each record
/// is compared with an independent run of exactly `s` steps (CUDA for
/// positions and energies; CPU for the virial, which an unobserved CUDA run
/// does not evaluate). Sub-second, tmp files only.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/pressure_series.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/record/pressure_series.cuh"
#include "atomiverse/cuda/record/recorder.cuh"
#include "atomiverse/cuda/record/thermo.cuh"

using namespace ATV;

namespace {

constexpr int kN = 10;
constexpr double kEps = 0.0104; // [eV]
constexpr double kSigma = 3.4;  // [Å]
constexpr double kCutoff = 6.0; // [Å]
constexpr double kSkin = 0.3;   // [Å]
constexpr double kDt = 1.0;     // [fs]
constexpr double kBox = 30.0;   // [Å]

/// @brief Ten argon atoms 3.5 Å apart (repulsive: energies and the virial
///        move every step) with a deterministic velocity field; no RNG.
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

void mount_cuda_lj(cuda::md::MD& sim) {
    sim.on(md::Phase::force)
        .add(cuda::NeighborList{kCutoff, kSkin}, -10)
        .add(cuda::PairLJCut{kEps, kSigma, kCutoff});
    sim.set_integrator(cuda::md::VelocityVerlet{kDt});
}

/// @brief CUDA oracle: the fixture after exactly `steps` steps of a fresh
///        driver with no recorder.
cpu::Frame cuda_state_after(int steps) {
    cpu::Frame frame = make_frame();
    cuda::md::MD sim;
    mount_cuda_lj(sim);
    sim.build();
    sim.run(frame, steps);
    return frame;
}

/// @brief CPU oracle (evaluates the pair virial every step).
cpu::Frame cpu_state_after(int steps) {
    cpu::Frame frame = make_frame();
    cpu::md::MD sim;
    sim.on(md::Phase::force).add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.build();
    sim.run(frame, steps);
    return frame;
}

std::string fresh_tmp(const char* name) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove(p, ec);
    return p.string();
}

void remove_file(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

/// @brief Data rows of a whitespace table, `#` lines skipped.
std::vector<std::vector<double>> read_rows(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        std::vector<double> row;
        double v = 0;
        while (ss >> v)
            row.push_back(v);
        if (!row.empty())
            rows.push_back(std::move(row));
    }
    return rows;
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
        std::getline(in, line); // comment
        std::vector<std::array<double, 3>> pos;
        for (int i = 0; i < n; ++i) {
            if (!std::getline(in, line))
                throw std::runtime_error("truncated XYZ frame");
            std::istringstream ss(line);
            std::string symbol;
            std::array<double, 3> r{};
            if (!(ss >> symbol >> r[0] >> r[1] >> r[2]))
                throw std::runtime_error("bad XYZ atom line: " + line);
            pos.push_back(r);
        }
        frames.push_back(std::move(pos));
    }
    return frames;
}

/// @brief Every written atom sits on a distinct oracle atom.
void expect_positions_are(const std::vector<std::array<double, 3>>& pos,
                          const cpu::Frame& ref) {
    const double tol = sizeof(cuda::pos_t) == 8 ? 1e-9 : 1e-5;
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
        EXPECT_FALSE(used[static_cast<std::size_t>(best)]) << "atom " << i;
        used[static_cast<std::size_t>(best)] = true;
        EXPECT_LT(best_d, tol) << "atom " << i << " is not the oracle state";
    }
}

} // namespace

// The oracle separates consecutive steps far beyond every tolerance below.
TEST(CudaRecorderStepConvention, OracleSeparatesConsecutiveSteps) {
    const cpu::Frame a = cuda_state_after(2), b = cuda_state_after(3);
    double dx = 0.0;
    for (int i = 0; i < kN; ++i)
        dx = std::max(dx, std::abs(a.span(kAtom_x)[i] - b.span(kAtom_x)[i]));
    EXPECT_GT(dx, 1e-4);
    const double ka = a.meta.get(kKinetic), kb = b.meta.get(kKinetic);
    EXPECT_GT(std::abs(ka - kb), 1e-5 * std::abs(ka));
}

// ── XyzRecorder ──────────────────────────────────────────────────────────

// A fresh 6-step run at every 3 writes the states after 0, 3 and 6 steps.
TEST(CudaRecorderStepConvention, XyzFreshRunWritesZeroThroughN) {
    const std::string path = fresh_tmp("atv_cuda_xyz_steps.xyz");
    {
        cpu::Frame frame = make_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.on(md::Phase::end_of_step).add(cuda::XyzRecorder{path, 3}, +10);
        sim.build();
        sim.run(frame, 6);
    } // teardown drains the callbacks
    const auto frames = read_xyz(path);
    ASSERT_EQ(frames.size(), 3u);
    const int want[] = {0, 3, 6};
    for (std::size_t f = 0; f < frames.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        expect_positions_are(frames[f], cuda_state_after(want[f]));
    }
    remove_file(path);
}

// Continued runs: the boundary state is written once (LAMMPS `last_dump`).
TEST(CudaRecorderStepConvention, XyzContinuedRunWritesTheBoundaryStepOnce) {
    const std::string path = fresh_tmp("atv_cuda_xyz_continued.xyz");
    {
        cpu::Frame frame = make_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.on(md::Phase::end_of_step).add(cuda::XyzRecorder{path, 2}, +10);
        sim.build();
        sim.run(frame, 4); // 0, 2, 4
        sim.run(frame, 4); // (4 already written), 6, 8
    }
    const auto frames = read_xyz(path);
    ASSERT_EQ(frames.size(), 5u);
    for (std::size_t f = 0; f < frames.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        expect_positions_are(frames[f],
                             cuda_state_after(static_cast<int>(2 * f)));
    }
    remove_file(path);
}

// ── Thermo ───────────────────────────────────────────────────────────────

namespace {

/// @brief Thermo `step pe ke` rows of a run over the chunks `runs`.
std::vector<std::vector<double>>
thermo_rows(int every, const std::vector<int>& runs, const std::string& path) {
    {
        cpu::Frame frame = make_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.on(md::Phase::end_of_step)
            .add(cuda::Thermo{every, {"step", "pe", "ke"}, path}, +20);
        sim.build();
        for (int n : runs)
            sim.run(frame, n);
    } // teardown drains the callbacks and closes the file
    return read_rows(path);
}

/// @brief Each row is the state after `row[0]` steps (8 significant
///        digits printed).
void expect_rows_are_oracle_states(
    const std::vector<std::vector<double>>& rows) {
    for (const auto& r : rows) {
        ASSERT_EQ(r.size(), 3u);
        SCOPED_TRACE("step " + std::to_string(r[0]));
        const cpu::Frame ref = cuda_state_after(static_cast<int>(r[0]));
        const double pe = ref.meta.get(kEnergy), ke = ref.meta.get(kKinetic);
        EXPECT_NEAR(r[1], pe, 1e-7 * std::abs(pe) + 1e-14);
        EXPECT_NEAR(r[2], ke, 1e-7 * std::abs(ke) + 1e-14);
    }
}

std::vector<long long> steps_of(const std::vector<std::vector<double>>& rows) {
    std::vector<long long> steps;
    for (const auto& r : rows)
        steps.push_back(static_cast<long long>(r[0]));
    return steps;
}

} // namespace

// A fresh 6-step run at every 2 prints steps 0, 2, 4, 6.
TEST(CudaRecorderStepConvention, ThermoFreshRunPrintsZeroThroughN) {
    const std::string path = fresh_tmp("atv_cuda_thermo_fresh.dat");
    const auto rows = thermo_rows(2, {6}, path);
    EXPECT_EQ(steps_of(rows), (std::vector<long long>{0, 2, 4, 6}));
    expect_rows_are_oracle_states(rows);
    remove_file(path);
}

// The last step of a run is printed off cadence (LAMMPS `next_thermo =
// MIN(next_thermo, update->laststep)`).
TEST(CudaRecorderStepConvention, ThermoPrintsTheLastStepOffCadence) {
    const std::string path = fresh_tmp("atv_cuda_thermo_last.dat");
    const auto rows = thermo_rows(2, {5}, path);
    EXPECT_EQ(steps_of(rows), (std::vector<long long>{0, 2, 4, 5}));
    expect_rows_are_oracle_states(rows);
    remove_file(path);
}

// Every run opens with the current state (LAMMPS `Output::setup` runs
// `thermo->compute(0)` on every run), so step 4 opens the second run too.
TEST(CudaRecorderStepConvention, ThermoContinuedRunOpensWithTheCurrentState) {
    const std::string path = fresh_tmp("atv_cuda_thermo_continued.dat");
    const auto rows = thermo_rows(2, {4, 4}, path);
    EXPECT_EQ(steps_of(rows), (std::vector<long long>{0, 2, 4, 4, 6, 8}));
    expect_rows_are_oracle_states(rows);
    remove_file(path);
}

// LAMMPS `thermo 0`: the first and the last step of every run only — the
// cadence never fires; the setup line and the reported final step do.
TEST(CudaRecorderStepConvention, ThermoCadenceZeroPrintsFirstAndLastStep) {
    const std::string path = fresh_tmp("atv_cuda_thermo_zero.dat");
    auto rows = thermo_rows(0, {6}, path);
    EXPECT_EQ(steps_of(rows), (std::vector<long long>{0, 6}));
    expect_rows_are_oracle_states(rows);
    rows = thermo_rows(0, {4, 4}, path);
    EXPECT_EQ(steps_of(rows), (std::vector<long long>{0, 4, 4, 8}));
    expect_rows_are_oracle_states(rows);
    remove_file(path);
}

// ── PressureSeries ───────────────────────────────────────────────────────

// A fresh 6-step run at every 2 writes rows for steps 0, 2, 4, 6, each the
// −P* of the state after that many steps: the CPU oracle's pair virial and
// lab-frame kinetic tensor through the shared row formula.
TEST(CudaRecorderStepConvention, PressureSeriesRowsAreTheStateAfterTheirStep) {
    const std::string path = fresh_tmp("atv_cuda_pressure_series.dat");
    {
        cpu::Frame frame = make_frame();
        cuda::md::MD sim;
        mount_cuda_lj(sim);
        sim.on(md::Phase::end_of_step)
            .add(cuda::md::PressureSeries{path, 2, kEps, 0}, +20);
        sim.build();
        sim.run(frame, 6);
    }
    const auto rows = read_rows(path);
    ASSERT_EQ(rows.size(), 4u);
    const double tol = sizeof(cuda::force_t) == 8 ? 1e-6 : 1e-3;
    double previous = 0.0;
    for (std::size_t f = 0; f < rows.size(); ++f) {
        const auto& r = rows[f];
        ASSERT_EQ(r.size(), 14u);
        const long long step = static_cast<long long>(r[0]);
        SCOPED_TRACE("row " + std::to_string(f) + " step "
                     + std::to_string(step));
        EXPECT_EQ(step, static_cast<long long>(2 * f));
        const cpu::Frame ref = cpu_state_after(static_cast<int>(step));
        double stress[6], kinetic[6] = {};
        for (int a = 0; a < 6; ++a)
            stress[a] = ref.meta.get(kStress)[a];
        for (int i = 0; i < kN; ++i) {
            const double m =
                ref.span(kAtom_mass)[i] * units::UnitSystem::lj().mvv2e;
            const double v[3] = {ref.span(kAtom_vx)[i], ref.span(kAtom_vy)[i],
                                 ref.span(kAtom_vz)[i]};
            kinetic[0] += m * v[0] * v[0];
            kinetic[1] += m * v[1] * v[1];
            kinetic[2] += m * v[2] * v[2];
            kinetic[3] += m * v[0] * v[1];
            kinetic[4] += m * v[0] * v[2];
            kinetic[5] += m * v[1] * v[2];
        }
        double cell[9] = {};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                cell[3 * a + b] = ref.box.cell()(a, b);
        cpu::PressureSeriesRow want;
        std::string error;
        ASSERT_TRUE(cpu::pressure_series_row(stress, kinetic, cell, cell, 0,
                                             kEps, step, want, error))
            << error;
        EXPECT_NEAR(r[1], 0.0, 1e-12) << "no deformation: strain stays 0";
        for (int c = 0; c < 6; ++c)
            EXPECT_NEAR(r[2 + c], want.minus_p_star[c],
                        tol * std::abs(want.minus_p_star[c]) + 1e-12)
                << "-P* component " << c;
        if (f > 0)
            EXPECT_GT(std::abs(want.minus_p_star[0] - previous),
                      10 * tol * std::abs(previous))
                << "the oracle must separate the sampled steps";
        previous = want.minus_p_star[0];
    }
    remove_file(path);
}
