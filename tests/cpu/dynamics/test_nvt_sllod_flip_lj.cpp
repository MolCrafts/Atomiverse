/// @file tests/cpu/dynamics/test_nvt_sllod_flip_lj.cpp
/// @brief A `cpu::md::NvtSllod` xy shear whose tilt crosses L/2: the flip
///        happens, and every step's LJ energy and forces equal those of a
///        freshly built neighbor list on the same configuration.
#include <array>
#include <cmath>
#include <cstdint>
#include <map>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nvt_sllod.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kEps = 0.0104, kSigma = 3.4, kCut = 7.0, kSkin = 1.0;

/// 5x5x5 lattice in fractional coordinates of a cell sheared to just below
/// the flip threshold.
cpu::Frame sheared_lattice() {
    const int side = 5;
    const double L = 19.0;
    Mat3 cell = Mat3::diagonal(L, L, L);
    cell(0, 1) = 0.49 * L;
    cpu::Frame f(side * side * side);
    int n = 0;
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k, ++n) {
                const double s[3] = {(i + 0.5) / side, (j + 0.5) / side,
                                     (k + 0.5) / side};
                f.span(kAtom_x)[n] = cell(0, 0) * s[0] + cell(0, 1) * s[1];
                f.span(kAtom_y)[n] = cell(1, 1) * s[1];
                f.span(kAtom_z)[n] = cell(2, 2) * s[2];
                // Deterministic small peculiar velocities [A/fs].
                f.span(kAtom_vx)[n] = 1e-3 * std::sin(1.3 * n);
                f.span(kAtom_vy)[n] = 1e-3 * std::cos(0.7 * n);
                f.span(kAtom_vz)[n] = 1e-3 * std::sin(2.1 * n + 0.4);
                f.span(kAtom_mass)[n] = 39.948;
                f.span(kAtom_type)[n] = 0;
                f.span(kAtom_tag)[n] = n + 1;
            }
    f.box = cpu::Box(cell);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    return f;
}

struct Snapshot {
    double energy = 0;
    std::map<std::int64_t, std::array<double, 3>> force;
};

/// Owned rows only, in `cell` (the cell the MD step evaluated forces in, or
/// a lattice-equivalent one), wrapped by the evaluation's own setup.
Snapshot fresh_evaluation(const cpu::Frame& md_frame, const Mat3& cell) {
    const int n = md_frame.nlocal();
    cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = md_frame.span(kAtom_x)[i];
        f.span(kAtom_y)[i] = md_frame.span(kAtom_y)[i];
        f.span(kAtom_z)[i] = md_frame.span(kAtom_z)[i];
        f.span(kAtom_mass)[i] = md_frame.span(kAtom_mass)[i];
        f.span(kAtom_type)[i] = md_frame.span(kAtom_type)[i];
        f.span(kAtom_tag)[i] = md_frame.span(kAtom_tag)[i];
    }
    f.box = cpu::Box(cell);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin, 1, 0, true});
    eval.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    eval.build();
    eval.run(f);
    Snapshot out;
    out.energy = f.meta.get(kEnergy);
    for (int i = 0; i < f.nlocal(); ++i)
        out.force[f.span(kAtom_tag)[i]] = {
            f.span(kAtom_fx)[i], f.span(kAtom_fy)[i], f.span(kAtom_fz)[i]};
    return out;
}

} // namespace

TEST(CpuNvtSllod, ShearAcrossTheFlipKeepsFreshListForcesAndEnergy) {
    cpu::Frame frame = sheared_lattice();
    const double L = 19.0;
    // xy grows by rate * L * dt = 0.076 per step from 9.31: past L/2 = 9.5
    // at the third end_of_step, flipped in the fourth step.
    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin, 1, 0, true});
    sim.set_integrator(
        cpu::md::NvtSllod{100.0, 100.0, 2.0, 0.002, md::kFlowXY, 3});
    sim.build();
    int flipped = -1;
    double previous_tilt = frame.box.cell()(0, 1);
    double previous_energy = 0;
    for (int step = 0; step < 8; ++step) {
        // Forces are evaluated before end_of_step deforms the cell; on the
        // flip step in the flipped cell, which is lattice-equivalent to this
        // unflipped one, so the reference also checks the flip itself.
        const Mat3 force_cell = frame.box.cell();
        sim.run(frame, 1);
        const double tilt = frame.box.cell()(0, 1);
        if (tilt < previous_tilt - 0.5 * L) {
            flipped = step;
            EXPECT_NEAR(tilt, previous_tilt + 0.076 - L, 1e-9);
        }
        previous_tilt = tilt;
        const Snapshot ref = fresh_evaluation(frame, force_cell);
        const double energy = frame.meta.get(kEnergy);
        EXPECT_NEAR(energy, ref.energy, 1e-10 * std::abs(ref.energy))
            << "step " << step;
        for (int i = 0; i < frame.nlocal(); ++i) {
            const auto& want = ref.force.at(frame.span(kAtom_tag)[i]);
            EXPECT_NEAR(frame.span(kAtom_fx)[i], want[0], 1e-10);
            EXPECT_NEAR(frame.span(kAtom_fy)[i], want[1], 1e-10);
            EXPECT_NEAR(frame.span(kAtom_fz)[i], want[2], 1e-10);
        }
        // Continuous across the flip: one step changes the energy little.
        if (step > 0)
            EXPECT_LT(std::abs(energy - previous_energy),
                      1e-2 * std::abs(previous_energy))
                << "step " << step;
        previous_energy = energy;
    }
    EXPECT_EQ(flipped, 3);
    EXPECT_LT(std::abs(frame.box.cell()(0, 1)), 0.5 * L);
}
