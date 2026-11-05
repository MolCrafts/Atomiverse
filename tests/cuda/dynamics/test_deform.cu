/// @file tests/cuda/dynamics/test_deform.cu
/// @brief cuda::md::Deform against cpu::md::Deform (erate / trate) on a
///        tilted cell: the cell row and the coordinates scale together, so
///        fractional coordinates are preserved.
#include <cmath>
#include <gtest/gtest.h>
#include <tuple>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/deform.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/deform.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;

namespace {

cpu::Frame tilted_input() {
    Mat3 cell = Mat3::diagonal(10.0, 9.0, 8.0);
    cell(0, 1) = 2.0;
    cell(0, 2) = -1.0;
    cell(1, 2) = 1.5;
    cpu::Frame f(4);
    const double x[4] = {1.0, 6.0, 9.0, 3.5};
    const double y[4] = {2.0, 8.0, 0.5, 4.0};
    const double z[4] = {7.0, 1.0, 3.0, 5.5};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = y[i];
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box(cell);
    return f;
}

class CudaDeform : public testing::TestWithParam<std::tuple<bool, int>> {};

} // namespace

TEST_P(CudaDeform, MatchesCpuCellRowAndCoordinates) {
    const auto [trate, axis] = GetParam();
    const double rate = 2e-3, dt = 1.0;
    const int steps = 3;

    cpu::Frame reference = tilted_input();
    const cpu::Frame input = tilted_input();
    {
        cpu::Context ctx;
        cpu::md::Deform deform(trate ? cpu::md::Deform::Mode::trate
                                     : cpu::md::Deform::Mode::erate,
                               rate, dt, axis);
        deform.build({ctx, reference});
        for (int s = 0; s < steps; ++s)
            deform.launch(
                {ctx, reference, slot_of(md::Phase::initial_integrate), s});
    }

    cpu::Frame device = tilted_input();
    {
        ATV::cuda::Frame frame = cpu::to_cuda(device);
        ATV::cuda::Context context;
        ATV::cuda::md::Deform deform(trate ? ATV::cuda::md::Deform::Mode::trate
                                           : ATV::cuda::md::Deform::Mode::erate,
                                     rate, dt, axis);
        deform.build({context, frame});
        for (int s = 0; s < steps; ++s)
            deform.launch(
                {context, frame, slot_of(md::Phase::initial_integrate), s});
        ATV_CUDA_CHECK(cudaDeviceSynchronize());
        ATV::cuda::to_cpu(frame, device);
        deform.destroy({context, frame});
        ATV::cuda::destroy(frame);
    }

    const double L0 = input.box.cell()(axis, axis);
    const double t = steps * dt;
    const double want = trate ? L0 * std::exp(rate * t) : L0 * (1 + rate * t);
    EXPECT_NEAR(reference.box.cell()(axis, axis), want, 1e-12);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            EXPECT_NEAR(device.box.cell()(a, b), reference.box.cell()(a, b),
                        1e-12)
                << "cell(" << a << "," << b << ")";
    for (int i = 0; i < 4; ++i) {
        const Vec3 r{device.span(kAtom_x)[i], device.span(kAtom_y)[i],
                     device.span(kAtom_z)[i]};
        const Vec3 r0{input.span(kAtom_x)[i], input.span(kAtom_y)[i],
                      input.span(kAtom_z)[i]};
        const auto s = device.box.to_fractional(r);
        const auto s0 = input.box.to_fractional(r0);
        for (int a = 0; a < 3; ++a)
            EXPECT_NEAR(s[a], s0[a], 1e-10) << "row " << i << " axis " << a;
        EXPECT_NEAR(device.span(kAtom_x)[i], reference.span(kAtom_x)[i], 1e-10);
        EXPECT_NEAR(device.span(kAtom_y)[i], reference.span(kAtom_y)[i], 1e-10);
        EXPECT_NEAR(device.span(kAtom_z)[i], reference.span(kAtom_z)[i], 1e-10);
    }
}

INSTANTIATE_TEST_SUITE_P(ErateTrateAxes, CudaDeform,
                         testing::Combine(testing::Bool(),
                                          testing::Values(0, 1)));
