/// @file tests/cuda/dynamics/test_nvt_sllod_flip_lj.cu
/// @brief An xy SLLOD shear whose tilt crosses L/2 under LJ + the cluster
///        neighbor list: the device flips the cell at the same step as the
///        CPU, and every step's energy and forces match the CPU run (whose
///        own fresh-list agreement is `test_nvt_sllod_flip_lj.cpp`).
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <type_traits>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nvt_sllod.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

constexpr bool kSingle = std::is_same_v<ATV::cuda::force_t, float>;
constexpr double kEnergyTol = kSingle ? 1e-4 : 1e-9;
constexpr double kForceTol = kSingle ? 1e-4 : 1e-8;
constexpr double kPosTol = kSingle ? 1e-4 : 1e-9;
constexpr double kEps = 0.0104, kSigma = 3.4, kCut = 7.0, kSkin = 1.0;
constexpr double kL = 19.0;

cpu::Frame sheared_lattice() {
    const int side = 5;
    Mat3 cell = Mat3::diagonal(kL, kL, kL);
    cell(0, 1) = 0.49 * kL;
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

struct Row {
    std::array<double, 3> unwrapped, force;
};

std::map<std::int64_t, Row> rows(const cpu::Frame& f) {
    std::map<std::int64_t, Row> out;
    const auto& h = f.box.cell();
    for (int i = 0; i < f.nlocal(); ++i) {
        Row r;
        const double x[3] = {f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                             f.span(kAtom_z)[i]};
        for (int a = 0; a < 3; ++a) {
            r.unwrapped[a] = x[a];
            for (int b = 0; b < 3; ++b)
                r.unwrapped[a] +=
                    h(a, b) * static_cast<double>(f.span(kAtom_image)(i, b));
        }
        r.force = {f.span(kAtom_fx)[i], f.span(kAtom_fy)[i],
                   f.span(kAtom_fz)[i]};
        out[f.span(kAtom_tag)[i]] = r;
    }
    return out;
}

} // namespace

TEST(CudaNvtSllodFlipLj, FlipStepEnergyAndForcesMatchCpu) {
    cpu::Frame cpu_frame = sheared_lattice();
    cpu::Frame gpu_frame = sheared_lattice();
    cpu::md::MD cpu_sim;
    cpu_sim.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    cpu_sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin, 1, 0, true});
    cpu_sim.set_integrator(
        cpu::md::NvtSllod{100.0, 100.0, 2.0, 0.002, md::kFlowXY, 3});
    cpu_sim.build();
    cuda::md::MD gpu_sim;
    gpu_sim.set_evaluator(cuda::PairLJCut{kEps, kSigma, kCut});
    gpu_sim.set_neighbor(cuda::NeighborList{kCut, kSkin});
    gpu_sim.set_integrator(
        cuda::md::NvtSllod{100.0, 100.0, 2.0, 0.002, md::kFlowXY, 3});
    gpu_sim.build();

    int flip_step = -1;
    double previous_tilt = gpu_frame.box.cell()(0, 1);
    for (int step = 0; step < 8; ++step) {
        cpu_sim.run(cpu_frame, 1);
        gpu_sim.run(gpu_frame, 1);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                ASSERT_NEAR(gpu_frame.box.cell()(a, b),
                            cpu_frame.box.cell()(a, b), 1e-12)
                    << "step " << step << " cell(" << a << "," << b << ")";
        const double tilt = gpu_frame.box.cell()(0, 1);
        if (tilt < previous_tilt - 0.5 * kL)
            flip_step = step;
        previous_tilt = tilt;
        const double e_cpu = cpu_frame.meta.get(kEnergy);
        const double e_gpu = gpu_frame.meta.get(kEnergy);
        EXPECT_NEAR(e_gpu, e_cpu, kEnergyTol * std::abs(e_cpu))
            << "step " << step;
        const auto want = rows(cpu_frame);
        const auto got = rows(gpu_frame);
        ASSERT_EQ(got.size(), want.size());
        for (const auto& [tag, w] : want) {
            const auto& g = got.at(tag);
            for (int a = 0; a < 3; ++a) {
                EXPECT_NEAR(g.unwrapped[a], w.unwrapped[a], kPosTol)
                    << "step " << step << " tag " << tag;
                EXPECT_NEAR(g.force[a], w.force[a], kForceTol)
                    << "step " << step << " tag " << tag;
            }
        }
    }
    // 9.31 + 0.076 k passes 9.5 at the third end_of_step; the device runs
    // the flip in the fourth step's pre_exchange.
    EXPECT_EQ(flip_step, 3);
    EXPECT_LT(std::abs(gpu_frame.box.cell()(0, 1)), 0.5 * kL);
}
