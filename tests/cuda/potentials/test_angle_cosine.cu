/// @file tests/cuda/potentials/test_angle_cosine.cu
/// @brief cuda::AngleCosine energy, forces and virial against the analytic
///        energy and the same-settings CPU AngleCosine.
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cpu/potentials/angle_cosine.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/potentials/angle_cosine.cuh"

using namespace ATV;

namespace {

constexpr double kTolerance =
    std::is_same_v<ATV::cuda::force_t, float> ? 5e-5 : 1e-11;
constexpr double kBox = 30.0;

// Three angles (60°, 120°, 150° at distinct arm lengths), two types, the
// third one wrapped across the periodic boundary so min-image is exercised.
cpu::Frame angles_frame() {
    constexpr int n = 9;
    cpu::Frame f(n);
    const double deg[3] = {60.0, 120.0, 150.0};
    const double r1[3] = {1.0, 1.2, 0.9};
    const double r2[3] = {1.1, 0.8, 1.3};
    const double cx[3] = {5.0, 15.0, 29.6};
    for (int a = 0; a < 3; ++a) {
        const double th = deg[a] * units::pi / 180.0;
        const double c[3] = {cx[a], 4.0 + a, 6.0};
        const double pi[3] = {c[0] + r1[a], c[1], c[2]};
        const double pk[3] = {c[0] + r2[a] * std::cos(th),
                              c[1] + r2[a] * std::sin(th) * 0.6,
                              c[2] + r2[a] * std::sin(th) * 0.8};
        const double* rows[3] = {pi, c, pk};
        for (int m = 0; m < 3; ++m) {
            const int row = 3 * a + m;
            // Wrap into [0, L): the third tuple straddles x = L.
            f.span(kAtom_x)[row] = std::fmod(rows[m][0] + kBox, kBox);
            f.span(kAtom_y)[row] = rows[m][1];
            f.span(kAtom_z)[row] = rows[m][2];
        }
    }
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(kBox);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    f.add_angle(0, 1, 2, 0);
    f.add_angle(3, 4, 5, 1);
    f.add_angle(6, 7, 8, 1);
    f.meta.get(kEnergy) = 0;
    f.meta.get(kStress).fill(0);
    return f;
}

struct DeviceInput {
    ATV::cuda::Frame frame;
    ATV::cuda::Context context;
    explicit DeviceInput(const cpu::Frame& source)
        : frame(cpu::to_cuda(source)) {
        ATV::cuda::geom_t cell[9] = {kBox, 0, 0, 0, kBox, 0, 0, 0, kBox};
        context.host_box.assign_cell(cell);
    }
    ~DeviceInput() { ATV::cuda::destroy(frame); }
};

} // namespace

TEST(CudaAngleCosine, MatchesAnalyticEnergyAndCpuForcesAndVirial) {
    const std::vector<double> K = {0.3, 1.7};
    cpu::Frame cpu_frame = angles_frame();
    cpu::Context cpu_ctx;
    cpu::AngleCosine cpu_angle(K);
    cpu_angle.build({cpu_ctx, cpu_frame});
    cpu_angle.launch({cpu_ctx, cpu_frame, 0, 0, true, true});
    cpu_angle.destroy({cpu_ctx, cpu_frame});

    cpu::Frame host = angles_frame();
    DeviceInput input(host);
    ATV::cuda::AngleCosine angle(K);
    ATV::cuda::BuildContext build{input.context, input.frame};
    angle.build(build);
    angle.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(input.frame, host);
    angle.destroy({input.context, input.frame});

    const double deg[3] = {60.0, 120.0, 150.0};
    const double analytic = K[0] * (1 + std::cos(deg[0] * units::pi / 180))
                            + K[1] * (1 + std::cos(deg[1] * units::pi / 180))
                            + K[1] * (1 + std::cos(deg[2] * units::pi / 180));
    EXPECT_NEAR(host.meta.get(kEnergy), analytic, kTolerance);
    EXPECT_NEAR(cpu_frame.meta.get(kEnergy), analytic, 1e-13);
    for (int i = 0; i < host.nlocal(); ++i) {
        EXPECT_NEAR(host.span(kAtom_fx)[i], cpu_frame.span(kAtom_fx)[i],
                    kTolerance)
            << "row " << i;
        EXPECT_NEAR(host.span(kAtom_fy)[i], cpu_frame.span(kAtom_fy)[i],
                    kTolerance)
            << "row " << i;
        EXPECT_NEAR(host.span(kAtom_fz)[i], cpu_frame.span(kAtom_fz)[i],
                    kTolerance)
            << "row " << i;
    }
    for (int c = 0; c < 6; ++c)
        EXPECT_NEAR(host.meta.get(kStress)[c], cpu_frame.meta.get(kStress)[c],
                    kTolerance)
            << "virial component " << c;
    // Not a vacuous comparison: the CPU forces and virial are non-zero. The
    // end-atom force is normal to its bond (here along x), so test y.
    EXPECT_GT(std::abs(cpu_frame.span(kAtom_fy)[0]), 1e-3);
    EXPECT_GT(std::abs(cpu_frame.meta.get(kStress)[0]), 1e-3);
}

TEST(CudaAngleCosine, UnrequestedEnergyAndVirialAreNotTouched) {
    cpu::Frame host = angles_frame();
    DeviceInput input(host);
    ATV::cuda::AngleCosine angle({0.3, 1.7});
    ATV::cuda::BuildContext build{input.context, input.frame};
    angle.build(build);
    ATV::cuda::LaunchContext ctx{input.context, input.frame, 0, 0};
    ctx.energy_requested = false;
    ctx.stress_requested = false;
    angle.launch(ctx);
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(input.frame, host);
    angle.destroy({input.context, input.frame});
    EXPECT_EQ(host.meta.get(kEnergy), 0.0);
    for (int c = 0; c < 6; ++c)
        EXPECT_EQ(host.meta.get(kStress)[c], 0.0);
    EXPECT_GT(std::abs(host.span(kAtom_fy)[0]), 1e-3);
}

TEST(CudaAngleCosine, BuildRejectsAnAngleTypeOutsideTheTable) {
    cpu::Frame host = angles_frame();
    DeviceInput input(host);
    ATV::cuda::AngleCosine angle({0.3}); // type 1 is used by two angles
    ATV::cuda::BuildContext build{input.context, input.frame};
    EXPECT_THROW(angle.build(build), std::invalid_argument);
    angle.destroy({input.context, input.frame});
    EXPECT_THROW(ATV::cuda::AngleCosine(std::vector<double>{}),
                 std::invalid_argument);
}
