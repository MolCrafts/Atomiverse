/// @file tests/cuda/potentials/test_harmonic_coefficients.cu
/// @brief Single bonded-component coefficient uploads and analytic force checks.
#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"

using namespace ATV;

namespace {

constexpr double kTolerance =
    std::is_same_v<ATV::cuda::work_t, float> ? 5e-5 : 1e-11;

void initialize(cpu::Frame& frame) {
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_tag)[i] = i;
        frame.span(kAtom_mass)[i] = 1;
    }
    frame.box = cpu::Box::cubic(20.0);
    frame.rebuild_tag_index(0, frame.nlocal());
}

// Component-author test harness: no Driver, MD step, or force-field solve.
struct DeviceInput {
    ATV::cuda::Frame frame;
    ATV::cuda::Context context;
    explicit DeviceInput(const cpu::Frame& source)
        : frame(cpu::to_cuda(source)) {
        for (int d = 0; d < 3; ++d) {
            context.host_box.cell[4 * d] = 20;
            context.host_box.inv_l[d] = ATV::cuda::real_t(1.0 / 20.0);
            context.host_box.half_l[d] = 10;
        }
    }
    ~DeviceInput() { ATV::cuda::destroy(frame); }
};

} // namespace

TEST(CudaHarmonicCoefficients, BondBuildUploadsEveryTypeInWorkPrecision) {
    cpu::Frame host(4);
    initialize(host);
    const double x[] = {1.0, 2.25, 5.0, 7.0};
    for (int i = 0; i < 4; ++i)
        host.span(kAtom_x)[i] = x[i];
    host.add_bond(0, 1, 0);
    host.add_bond(2, 3, 1);
    DeviceInput input(host);
    ATV::cuda::BondHarmonic bond({1.75, 4.5}, {0.9, 1.1});
    ATV::cuda::BuildContext build{input.context, input.frame};
    bond.build(build);
    bond.launch({input.context, input.frame, 0, 0});
    ATV::cuda::to_cpu(input.frame, host);
    bond.destroy({input.context, input.frame});

    const double d0 = 1.25 - 0.9, d1 = 2.0 - 1.1;
    EXPECT_NEAR(host.meta.get(kEnergy), 1.75 * d0 * d0 + 4.5 * d1 * d1,
                kTolerance);
    const double expected[] = {2 * 1.75 * d0, -2 * 1.75 * d0, 2 * 4.5 * d1,
                               -2 * 4.5 * d1};
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(host.span(kAtom_fx)[i], expected[i], kTolerance);
        EXPECT_NEAR(host.span(kAtom_fy)[i], 0.0, kTolerance);
        EXPECT_NEAR(host.span(kAtom_fz)[i], 0.0, kTolerance);
    }
}

TEST(CudaHarmonicCoefficients, AngleBuildUploadsEveryTypeInWorkPrecision) {
    cpu::Frame host(6);
    initialize(host);
    const double x[] = {1, 0, 0, 7, 5, 5};
    const double y[] = {0, 0, 1, 0, 0, 0.5};
    for (int i = 0; i < 6; ++i) {
        host.span(kAtom_x)[i] = x[i];
        host.span(kAtom_y)[i] = y[i];
    }
    host.add_angle(0, 1, 2, 0);
    host.add_angle(3, 4, 5, 1);
    DeviceInput input(host);
    ATV::cuda::AngleHarmonic angle({1.75, 4.5}, {1.1, 2.2});
    ATV::cuda::BuildContext build{input.context, input.frame};
    angle.build(build);
    angle.launch({input.context, input.frame, 0, 0});
    ATV::cuda::to_cpu(input.frame, host);
    angle.destroy({input.context, input.frame});

    const double d0 = units::pi / 2 - 1.1, d1 = units::pi / 2 - 2.2;
    EXPECT_NEAR(host.meta.get(kEnergy), 1.75 * d0 * d0 + 4.5 * d1 * d1,
                kTolerance);
    const double g0 = 2 * 1.75 * d0, g1 = 2 * 4.5 * d1;
    const double expected_x[] = {0, -g0, g0, 0, -2 * g1, 2 * g1};
    const double expected_y[] = {g0, -g0, 0, g1 / 2, -g1 / 2, 0};
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(host.span(kAtom_fx)[i], expected_x[i], kTolerance);
        EXPECT_NEAR(host.span(kAtom_fy)[i], expected_y[i], kTolerance);
        EXPECT_NEAR(host.span(kAtom_fz)[i], 0.0, kTolerance);
    }
}
