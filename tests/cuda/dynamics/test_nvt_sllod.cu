/// @file tests/cuda/dynamics/test_nvt_sllod.cu
/// @brief cuda::md::NvtSllod against the same-settings CPU class: one step
///        (positions, velocities, image flags, cell) for an edge and a shear
///        direction; a tchain = 3 run (chain variables, unwrapped state);
///        and a shear across the flip threshold (device-decided flip).
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/core/dynamics_math.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/nvt_sllod.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;

namespace {

constexpr double kTol =
    std::is_same_v<ATV::cuda::mixed_t, float> ? 1e-4 : 1e-11;
constexpr int kN = 6;

cpu::Frame sheared_input(const Mat3& cell = Mat3::diagonal(10.0, 9.0, 8.0)) {
    cpu::Frame f(kN);
    const double x[kN] = {0.5, 3.0, 9.9, 5.0, 7.5, 2.0};
    const double y[kN] = {8.95, 2.0, 4.0, 0.05, 6.0, 7.0};
    const double z[kN] = {1.0, 7.9, 3.0, 4.0, 0.1, 5.5};
    const double vx[kN] = {0.01, -0.02, 0.15, 0.0, 0.03, -0.01};
    const double vy[kN] = {0.12, 0.01, -0.02, -0.1, 0.0, 0.02};
    const double vz[kN] = {0.0, 0.14, 0.01, -0.03, -0.12, 0.0};
    for (int i = 0; i < kN; ++i) {
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = y[i];
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_vx)[i] = vx[i];
        f.span(kAtom_vy)[i] = vy[i];
        f.span(kAtom_vz)[i] = vz[i];
        f.span(kAtom_mass)[i] = 1.0 + 0.5 * i;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box(cell);
    return f;
}

constexpr md::Phase kStep[] = {
    md::Phase::initial_integrate, md::Phase::pre_exchange,
    md::Phase::final_integrate, md::Phase::end_of_step};

/// The CPU class runs pre_exchange every step here, as on a reneighbor
/// step, so it wraps rows like the CUDA class does inside its move.
md::SllodChainState cpu_run(cpu::Frame& f, int direction, double rate,
                            int steps, int tchain) {
    cpu::Context ctx;
    ctx.md.n_dof_global = md::degrees_of_freedom(f.nlocal());
    cpu::md::NvtSllod sllod(300.0, 50.0, 1.0, rate, direction, tchain);
    sllod.build({ctx, f});
    for (int s = 0; s < steps; ++s)
        for (auto phase : kStep)
            sllod.launch({ctx, f, slot_of(phase), s});
    return sllod.thermostat_chain();
}

md::SllodChainState cuda_run(cpu::Frame& host, int direction, double rate,
                             int steps, int tchain) {
    ATV::cuda::Frame frame = cpu::to_cuda(host);
    ATV::cuda::Context context;
    context.md.n_dof_global = md::degrees_of_freedom(host.nlocal());
    ATV::cuda::geom_t cell[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell[3 * i + j] = host.box.cell()(i, j);
    context.host_box.assign_cell(cell);
    ATV::cuda::md::NvtSllod sllod(300.0, 50.0, 1.0, rate, direction, tchain);
    sllod.build({context, frame});
    for (int s = 0; s < steps; ++s)
        for (auto phase : kStep)
            sllod.launch({context, frame, slot_of(phase), s});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    auto chain = sllod.thermostat_chain();
    ATV::cuda::to_cpu(frame, host);
    sllod.destroy({context, frame});
    ATV::cuda::destroy(frame);
    return chain;
}

/// r + H image, one axis.
double unwrapped(const cpu::Frame& f, int i, int a) {
    double u = a == 0   ? f.span(kAtom_x)[i]
               : a == 1 ? f.span(kAtom_y)[i]
                        : f.span(kAtom_z)[i];
    for (int b = 0; b < 3; ++b)
        u +=
            f.box.cell()(a, b) * static_cast<double>(f.span(kAtom_image)(i, b));
    return u;
}

void expect_same_state(const cpu::Frame& device, const cpu::Frame& reference,
                       double tol) {
    for (int i = 0; i < kN; ++i) {
        for (int a = 0; a < 3; ++a)
            EXPECT_NEAR(unwrapped(device, i, a), unwrapped(reference, i, a),
                        tol)
                << "row " << i << " axis " << a;
        EXPECT_NEAR(device.span(kAtom_vx)[i], reference.span(kAtom_vx)[i], tol)
            << "row " << i;
        EXPECT_NEAR(device.span(kAtom_vy)[i], reference.span(kAtom_vy)[i], tol)
            << "row " << i;
        EXPECT_NEAR(device.span(kAtom_vz)[i], reference.span(kAtom_vz)[i], tol)
            << "row " << i;
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(device.span(kAtom_image)(i, d),
                      reference.span(kAtom_image)(i, d))
                << "row " << i << " axis " << d;
    }
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            EXPECT_NEAR(device.box.cell()(a, b), reference.box.cell()(a, b),
                        1e-12)
                << "cell(" << a << "," << b << ")";
}

class CudaNvtSllodStep : public testing::TestWithParam<int> {};

} // namespace

TEST_P(CudaNvtSllodStep, MatchesCpuPositionsVelocitiesImagesAndCell) {
    const int direction = GetParam();
    // A large rate so the streaming profile moves rows across faces in one
    // step (rows 0, 2, 3 and 4 sit within 0.1 Å of a face).
    const double rate = 0.02;
    cpu::Frame reference = sheared_input();
    cpu_run(reference, direction, rate, 1, 1);
    cpu::Frame device = sheared_input();
    cuda_run(device, direction, rate, 1, 1);

    int crossings = 0;
    for (int i = 0; i < kN; ++i) {
        EXPECT_NEAR(device.span(kAtom_x)[i], reference.span(kAtom_x)[i], kTol)
            << "row " << i;
        EXPECT_NEAR(device.span(kAtom_y)[i], reference.span(kAtom_y)[i], kTol)
            << "row " << i;
        EXPECT_NEAR(device.span(kAtom_z)[i], reference.span(kAtom_z)[i], kTol)
            << "row " << i;
        for (int d = 0; d < 3; ++d)
            crossings += std::abs(int(reference.span(kAtom_image)(i, d)));
    }
    expect_same_state(device, reference, kTol);
    EXPECT_GT(crossings, 0) << "no row crossed a face; images untested";
    // The deformed slot actually moved.
    const int ab[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
    const double h0[6] = {10.0, 9.0, 8.0, 0.0, 0.0, 0.0};
    EXPECT_NE(reference.box.cell()(ab[direction][0], ab[direction][1]),
              h0[direction]);
}

// x edge (box grows, symmetric origin shift) and xy shear (tilt grows).
INSTANTIATE_TEST_SUITE_P(EdgeAndShear, CudaNvtSllodStep, testing::Values(0, 5));

class CudaNvtSllodChain : public testing::TestWithParam<int> {};

TEST_P(CudaNvtSllodChain, ThreeLinkChainMatchesCpuAfterEightSteps) {
    const int direction = GetParam();
    const double rate = 0.005;
    cpu::Frame reference = sheared_input();
    const auto want = cpu_run(reference, direction, rate, 8, 3);
    cpu::Frame device = sheared_input();
    const auto got = cuda_run(device, direction, rate, 8, 3);
    ASSERT_EQ(got.eta.size(), 3u);
    ASSERT_EQ(want.eta.size(), 3u);
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(got.eta[k], want.eta[k],
                    1e3 * kTol * (1e-6 + std::abs(want.eta[k])))
            << "link " << k;
        EXPECT_NEAR(got.eta_dot[k], want.eta_dot[k],
                    1e3 * kTol * (1e-9 + std::abs(want.eta_dot[k])))
            << "link " << k;
    }
    EXPECT_NE(want.eta_dot[2], 0.0) << "the outer link must move";
    expect_same_state(device, reference, 10 * kTol);
}

INSTANTIATE_TEST_SUITE_P(EdgeAndShear, CudaNvtSllodChain,
                         testing::Values(0, 3, 5));

TEST(CudaNvtSllod, DeviceDecidedFlipMatchesCpu) {
    // xy from 4.8 on lx = 10 grows 0.09 per step: decided at the third
    // end_of_step, executed at the fourth pre_exchange on both backends.
    Mat3 cell = Mat3::diagonal(10.0, 9.0, 8.0);
    cell(0, 1) = 4.8;
    for (int steps : {3, 4, 6}) {
        cpu::Frame reference = sheared_input(cell);
        cpu_run(reference, md::kFlowXY, 0.01, steps, 1);
        cpu::Frame device = sheared_input(cell);
        cuda_run(device, md::kFlowXY, 0.01, steps, 1);
        const double tilt = 4.8 + 0.09 * steps - (steps > 3 ? 10.0 : 0.0);
        EXPECT_NEAR(reference.box.cell()(0, 1), tilt, 1e-12) << steps;
        EXPECT_NEAR(device.box.cell()(0, 1), tilt, 1e-12) << steps;
        expect_same_state(device, reference, 10 * kTol);
    }
}
